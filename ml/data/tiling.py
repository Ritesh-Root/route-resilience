"""Deterministic raster/mask tiling with scene-aware 80/10/10 splits (DATA side).

Large satellite scenes (DeepGlobe / SpaceNet tiles are typically 1024-2048 px or
larger) must be cut into fixed 512x512 chips before they fit on a 4GB GPU. The
subtle part is the train/val/test split: tiles cut from the *same* scene share
context (roads cross tile borders, lighting/sensor signature is identical), so a
random per-tile split leaks information from train into val/test and inflates
metrics. We therefore split **by source scene**, never per tile — every chip from
one scene lands entirely in one split.

The split is fully deterministic: a scene's bucket is decided by a stable hash of
its scene id (+ a salt), so re-running on the same scene set always reproduces the
identical partition with no state file to keep in sync. This pairs with the
training-time occlusion augmentation in ``backend/app/services/occlusion.py``
(apply occlusion to the image chip, keep the clean mask chip as the target).

Heavy / optional deps (``rasterio`` for GeoTIFF I/O, ``skimage`` for resizing) are
imported lazily inside functions and guarded with a clear message, so this module
always imports even on a bare interpreter. The core tiling math is pure numpy.

Usage (no datasets are downloaded here — point ``--images`` at your own GeoTIFFs)::

    python ml/data/tiling.py \
        --images /path/to/scenes/*_sat.tif \
        --masks  /path/to/scenes/*_mask.tif \
        --out    ml/data/chips \
        --tile 512 --overlap 64

TODO(you): wire the produced manifest.csv into the training Dataset (each row is
one chip: image_path, mask_path, scene_id, split, row, col).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

import numpy as np

# 4GB-laptop-GPU friendly defaults.
TILE_SIZE: int = 512
DEFAULT_OVERLAP: int = 0
# 80/10/10. Buckets are cumulative fractions over a stable per-scene hash in [0,1).
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}

Split = Literal["train", "val", "test"]


# --------------------------------------------------------------------------- #
# Deterministic, scene-aware split assignment (NO spatial leakage).
# --------------------------------------------------------------------------- #
def scene_to_unit_interval(scene_id: str, salt: str = "route-resilience") -> float:
    """Map a scene id to a stable float in [0, 1) via SHA-256.

    Deterministic across runs, machines and Python's hash-randomization (unlike
    builtin ``hash``). Used to place an *entire scene* into one split bucket.
    """
    digest = hashlib.sha256(f"{salt}:{scene_id}".encode("utf-8")).digest()
    # First 8 bytes -> uint64 -> normalise to [0, 1).
    val = int.from_bytes(digest[:8], "big")
    return val / float(1 << 64)


def assign_split(scene_id: str,
                 fractions: dict[str, float] = SPLIT_FRACTIONS,
                 salt: str = "route-resilience") -> Split:
    """Assign a scene to 'train' | 'val' | 'test' deterministically.

    Splitting by *scene* (not per tile) is what prevents spatial leakage: all
    chips from one scene share the same bucket, so no road segment appears in
    both train and val/test.
    """
    u = scene_to_unit_interval(scene_id, salt=salt)
    cum = 0.0
    # Iterate in a fixed order so bucket edges are stable.
    for split in ("train", "val", "test"):
        cum += fractions[split]
        if u < cum:
            return split  # type: ignore[return-value]
    return "test"  # numerical safety net for u extremely close to 1.0


def scene_id_from_path(path: str) -> str:
    """Derive a scene id from a file path.

    Strips directory and a trailing role suffix like ``_sat`` / ``_mask`` / the
    extension so the image and its mask map to the *same* scene id. Adjust the
    suffix list if your dataset uses a different naming convention.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("_sat", "_satellite", "_image", "_img", "_mask", "_label", "_gt"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# --------------------------------------------------------------------------- #
# Pure-numpy tiling geometry.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileSpec:
    """Where a single chip lives inside its source scene (top-left + size)."""
    row: int
    col: int
    y0: int
    x0: int
    size: int


def tile_offsets(h: int, w: int, size: int = TILE_SIZE,
                 overlap: int = DEFAULT_OVERLAP) -> list[TileSpec]:
    """Compute deterministic top-left offsets covering an (h, w) scene.

    The grid steps by ``size - overlap``. The final row/column is clamped so the
    last chip's edge sits flush with the scene border (full coverage, last tiles
    may overlap their neighbour). Smaller-than-tile scenes yield a single chip at
    (0, 0) — pad it to ``size`` with :func:`pad_to`.
    """
    if size <= 0:
        raise ValueError("tile size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap

    def axis_starts(extent: int) -> list[int]:
        if extent <= size:
            return [0]
        starts = list(range(0, extent - size + 1, step))
        if starts[-1] != extent - size:
            starts.append(extent - size)  # clamp final tile flush to the edge
        return starts

    specs: list[TileSpec] = []
    for r, y0 in enumerate(axis_starts(h)):
        for c, x0 in enumerate(axis_starts(w)):
            specs.append(TileSpec(row=r, col=c, y0=y0, x0=x0, size=size))
    return specs


def pad_to(chip: np.ndarray, size: int = TILE_SIZE,
           fill: float = 0.0) -> np.ndarray:
    """Pad a chip on the bottom/right up to (size, size[, C]) — for edge scenes."""
    h, w = chip.shape[:2]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return chip
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (chip.ndim - 2)
    return np.pad(chip, pad_width, mode="constant", constant_values=fill)


def iter_tiles(image: np.ndarray, mask: np.ndarray | None = None,
               size: int = TILE_SIZE, overlap: int = DEFAULT_OVERLAP,
               ) -> Iterator[tuple[TileSpec, np.ndarray, np.ndarray | None]]:
    """Yield ``(spec, image_chip, mask_chip)`` chips, each padded to ``size``.

    ``image`` is (H, W) or (H, W, C); ``mask`` (if given) must share H, W. Both
    are sliced with the *same* offsets so chips stay pixel-aligned.
    """
    h, w = image.shape[:2]
    if mask is not None and mask.shape[:2] != (h, w):
        raise ValueError(
            f"image {image.shape[:2]} and mask {mask.shape[:2]} differ in H/W"
        )
    for spec in tile_offsets(h, w, size=size, overlap=overlap):
        ys, xs = slice(spec.y0, spec.y0 + size), slice(spec.x0, spec.x0 + size)
        img_chip = pad_to(image[ys, xs], size=size)
        mask_chip = pad_to(mask[ys, xs], size=size) if mask is not None else None
        yield spec, img_chip, mask_chip


# --------------------------------------------------------------------------- #
# Guarded raster I/O (rasterio optional; numpy .npy fallback always available).
# --------------------------------------------------------------------------- #
def _read_raster(path: str) -> np.ndarray:
    """Read a raster to a numpy array (H, W) or (H, W, C).

    GeoTIFFs go through ``rasterio`` (bands -> last axis). ``.npy`` files are read
    directly with numpy so the pipeline is testable without GDAL/rasterio.
    """
    if path.lower().endswith(".npy"):
        return np.load(path)
    try:
        import rasterio  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"Reading '{path}' needs rasterio (pip install rasterio), or convert "
            "the scene to a .npy array first. See ml/requirements.txt."
        ) from exc
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    return np.transpose(arr, (1, 2, 0)) if arr.shape[0] > 1 else arr[0]


def _save_chip(path: str, chip: np.ndarray) -> None:
    """Persist a chip. PNG via skimage when available, else .npy fallback."""
    if path.lower().endswith(".npy"):
        np.save(path, chip)
        return
    try:
        from skimage.io import imsave  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"Saving '{path}' as an image needs scikit-image "
            "(pip install scikit-image), or use a .npy output path instead."
        ) from exc
    imsave(path, chip.astype(np.uint8), check_contrast=False)


# --------------------------------------------------------------------------- #
# Orchestration: pair images<->masks, split by scene, write chips + manifest.
# --------------------------------------------------------------------------- #
@dataclass
class ManifestRow:
    scene_id: str
    split: Split
    row: int
    col: int
    image_path: str
    mask_path: str | None


def _pair_scenes(image_paths: Iterable[str],
                 mask_paths: Iterable[str]) -> list[tuple[str, str, str | None]]:
    """Pair each image with its mask by shared scene id. Returns
    ``[(scene_id, image_path, mask_path|None)]``."""
    masks_by_scene = {scene_id_from_path(m): m for m in mask_paths}
    pairs: list[tuple[str, str, str | None]] = []
    for img in sorted(image_paths):
        sid = scene_id_from_path(img)
        pairs.append((sid, img, masks_by_scene.get(sid)))
    return pairs


def build_chips(image_paths: list[str], mask_paths: list[str], out_dir: str,
                size: int = TILE_SIZE, overlap: int = DEFAULT_OVERLAP,
                ext: str = "npy", salt: str = "route-resilience",
                ) -> list[ManifestRow]:
    """Tile every scene, route chips into split subdirs, return manifest rows.

    Layout: ``out_dir/<split>/images/<scene>_r<row>_c<col>.<ext>`` (+ matching
    ``masks/``). Writing actual files needs rasterio/skimage unless ``ext='npy'``
    and inputs are ``.npy`` (the dependency-free path used by ``--self-test``).
    """
    rows: list[ManifestRow] = []
    for sid, img_path, mask_path in _pair_scenes(image_paths, mask_paths):
        split = assign_split(sid, salt=salt)
        image = _read_raster(img_path)
        mask = _read_raster(mask_path) if mask_path else None
        img_out = os.path.join(out_dir, split, "images")
        msk_out = os.path.join(out_dir, split, "masks")
        os.makedirs(img_out, exist_ok=True)
        if mask is not None:
            os.makedirs(msk_out, exist_ok=True)
        for spec, img_chip, mask_chip in iter_tiles(image, mask, size, overlap):
            name = f"{sid}_r{spec.row}_c{spec.col}.{ext}"
            ip = os.path.join(img_out, name)
            _save_chip(ip, img_chip)
            mp: str | None = None
            if mask_chip is not None:
                mp = os.path.join(msk_out, name)
                _save_chip(mp, mask_chip)
            rows.append(ManifestRow(sid, split, spec.row, spec.col, ip, mp))
    return rows


def write_manifest(rows: list[ManifestRow], out_dir: str) -> str:
    """Write ``manifest.csv`` (one row per chip) and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene_id", "split", "row", "col", "image_path", "mask_path"])
        for r in rows:
            writer.writerow([r.scene_id, r.split, r.row, r.col, r.image_path,
                             r.mask_path or ""])
    return path


def split_summary(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    """Per-split counts of {scenes, chips} — handy for a sanity log line."""
    out: dict[str, dict[str, int]] = {
        s: {"scenes": 0, "chips": 0} for s in ("train", "val", "test")
    }
    seen: set[tuple[str, str]] = set()
    for r in rows:
        out[r.split]["chips"] += 1
        if (r.split, r.scene_id) not in seen:
            seen.add((r.split, r.scene_id))
            out[r.split]["scenes"] += 1
    return out


# --------------------------------------------------------------------------- #
# CLI + dependency-free self-test (proves no spatial leakage by construction).
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run an in-memory check requiring only numpy — no I/O, no heavy deps."""
    # Tiling geometry: a 1300x1000 scene at 512/overlap-64 fully covers the edges.
    specs = tile_offsets(1300, 1000, size=512, overlap=64)
    assert max(s.y0 for s in specs) + 512 == 1300, "rows must reach bottom edge"
    assert max(s.x0 for s in specs) + 512 == 1000, "cols must reach right edge"

    # Padding a small scene yields exactly one full-size chip.
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    chips = list(iter_tiles(img, None, size=512))
    assert len(chips) == 1 and chips[0][1].shape == (512, 512, 3)

    # Split is deterministic and stable across calls.
    scenes = [f"scene_{i:04d}" for i in range(1000)]
    a = {s: assign_split(s) for s in scenes}
    b = {s: assign_split(s) for s in scenes}
    assert a == b, "split must be deterministic"

    # Fractions are roughly 80/10/10 on a large scene set.
    from collections import Counter
    counts = Counter(a.values())
    assert 0.74 < counts["train"] / len(scenes) < 0.86, counts
    assert 0.05 < counts["val"] / len(scenes) < 0.16, counts
    assert 0.05 < counts["test"] / len(scenes) < 0.16, counts

    # No leakage: every scene lands in exactly one split.
    placements = {s: assign_split(s) for s in scenes}
    assert len(set(placements)) == len(scenes)
    print("[tiling] self-test OK:",
          dict(scenes=len(scenes), split=dict(counts), tiles_per_scene=len(specs)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[],
                    help="glob(s) for source image rasters (GeoTIFF or .npy)")
    ap.add_argument("--masks", nargs="*", default=[],
                    help="glob(s) for source mask rasters (matched by scene id)")
    ap.add_argument("--out", default="ml/data/chips", help="output root dir")
    ap.add_argument("--tile", type=int, default=TILE_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--ext", default="npy", choices=("npy", "png"),
                    help="chip output format ('png' needs scikit-image)")
    ap.add_argument("--salt", default="route-resilience",
                    help="hash salt; keep fixed to reproduce the same split")
    ap.add_argument("--self-test", action="store_true",
                    help="run dependency-free correctness checks and exit")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    image_paths = [p for g in args.images for p in glob.glob(g)]
    mask_paths = [p for g in args.masks for p in glob.glob(g)]
    if not image_paths:
        ap.error("no --images matched; pass globs to your own rasters "
                 "(this script never downloads datasets) or use --self-test")

    rows = build_chips(image_paths, mask_paths, args.out, size=args.tile,
                       overlap=args.overlap, ext=args.ext, salt=args.salt)
    manifest = write_manifest(rows, args.out)
    print(f"[tiling] wrote {len(rows)} chips -> {manifest}")
    print(f"[tiling] split summary: {split_summary(rows)}")


if __name__ == "__main__":
    main()
