"""SpaceNet SN3 / SN5 centerline dataset skeleton — road extraction (ML side).

This module provides a **loader skeleton + notes** for the SpaceNet Roads
challenges. It is deliberately a scaffold: it must import cleanly on the demo
machine (Python 3.14, 4 GB RTX 3050) *without* any of the heavy geo/DL stack
installed and *without* downloading the multi-GB SpaceNet corpus. All heavy
imports (``torch``, ``rasterio``, ``shapely``, ``cv2``) are guarded so that
``import ml.data.spacenet`` always succeeds; the libs are only required when you
actually instantiate / iterate a dataset.

------------------------------------------------------------------------------
WHY SpaceNet SN3 / SN5 (centerlines), and what APLS measures
------------------------------------------------------------------------------
SpaceNet Roads ships **vector road centerlines** (LineStrings) rather than a
filled road *footprint* polygon:

  * **SN3 — Road Network Detection** (Las Vegas, Paris, Shanghai, Khartoum):
    8-band WorldView-3 + 3-band RGB tiles, ground-truth ``geojson_roads``
    LineStrings with a ``road_type`` / lane-count attribute.
  * **SN5 — Road Network + Routing/Speed** (Moscow, Mumbai, + SN3 cities):
    adds a per-segment **speed limit** attribute, so the extracted graph can be
    weighted by *travel time* — exactly what our backend needs for the
    ``travelTimeSec`` edge property and the resilience simulation.

The grader for these challenges is **APLS** (Average Path Length Similarity).
Unlike pixel IoU/Dice (which reward overlap but are blind to *connectivity*),
APLS compares the *graphs*: it injects equally-spaced midpoints along edges of
the ground-truth graph G_gt and the proposed graph G_prop, snaps each node onto
the other graph, and compares **shortest-path lengths** between every node pair:

    APLS = 1 - mean( min(1, |L_gt(a,b) - L_prop(a,b)| / L_gt(a,b)) )

A missing bridge or a single broken link blows up many shortest paths and
*tanks* APLS even when pixel IoU barely moves — which is the entire thesis of
this project: occlusion (cloud / shadow / tree canopy) silently severs links,
fragmenting the routing graph. We therefore train two heads on the **same**
labels and report **both** pixel metrics (IoU/Dice) and graph metrics
(APLS / connectivityRatio), so the demo can show the baseline losing
connectivity under occlusion while the robust head recovers it.

------------------------------------------------------------------------------
Centerline -> training label
------------------------------------------------------------------------------
The raw label is a vector LineString graph; a segmentation network needs a
raster target. Two standard rasterizations (we expose ``label_mode``):

  * ``"buffer"``  — buffer each centerline by ~ ``road_width_m`` (often derived
    from lane count) and rasterize to a binary road mask. Trains a plain
    semantic-segmentation U-Net; recover the graph at inference by
    skeleton
    -> spur-pruning -> graph build. Good IoU, lossy APLS.
  * ``"centerline"`` — rasterize a thin (1–2 px) centerline ribbon, optionally
    with a soft distance falloff. Pairs well with an APLS-friendly loss and a
    direct skeleton->graph decode. Lower IoU, better APLS.

Either way we keep the *vector* ground truth around so APLS can be computed on
the decoded graph at eval time (see ``ml/metrics/apls.py`` — TODO, not in this
file's scope).

------------------------------------------------------------------------------
Occlusion robustness hook
------------------------------------------------------------------------------
Robustness is injected on the **input chip only** (the target mask stays the
clean ground truth) using the synthetic occlusion generators in
``backend/app/services/occlusion.py`` (``add_clouds`` / ``add_shadows`` /
``add_canopy`` / ``random_occlude``) — numpy-only, so no extra CV deps. For the
strongest result, return *both* a clean and an occluded view of the same chip
and add a consistency loss between their predictions during training
(see ``ml/train/`` in the plan). We re-import those generators lazily here to
avoid coupling the ML package to the FastAPI app at import time.

------------------------------------------------------------------------------
DATA ACCESS (NO DOWNLOAD ON THE DEMO MACHINE)
------------------------------------------------------------------------------
SpaceNet is a Registry-of-Open-Data bucket: ``s3://spacenet-dataset`` (us-east-1,
requester-pays = NO, but it is many GB — DO NOT pull it here). Layout, e.g.::

    s3://spacenet-dataset/spacenet/SN3_roads/train/
        AOI_2_Vegas/PS-RGB/SN3_roads_train_AOI_2_Vegas_PS-RGB_img1.tif
        AOI_2_Vegas/geojson_roads/SN3_roads_train_AOI_2_Vegas_geojson_roads_img1.geojson
    s3://spacenet-dataset/spacenet/SN5_roads/train/AOI_7_Moscow/...

TODO (run on a beefy box / cloud, NOT here)::

    # list only (cheap), to confirm layout:
    aws s3 ls --no-sign-request s3://spacenet-dataset/spacenet/SN3_roads/train/

    # download ONE small AOI to a scratch dir for a smoke test:
    aws s3 sync --no-sign-request \
        s3://spacenet-dataset/spacenet/SN3_roads/train/AOI_2_Vegas \
        "/scratch/spacenet/SN3/AOI_2_Vegas"

Then point ``SpaceNetCenterlineDataset(root="/scratch/spacenet/SN3/AOI_2_Vegas")``
at the local copy. This file never touches S3 or the network.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

# numpy is the one light dependency we assume is present (the API already uses
# it). Import lazily-but-safely so the module still imports if it is missing.
try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy expected, but stay import-safe
    np = None  # type: ignore[assignment]

# Type-only imports never execute at runtime, so they cannot break ``import``.
if TYPE_CHECKING:  # pragma: no cover
    import numpy as _np  # noqa: F401
    from torch.utils.data import Dataset as _TorchDataset
else:
    _TorchDataset = object  # runtime base class fallback (no torch needed)


LabelMode = Literal["buffer", "centerline"]
SpaceNetChallenge = Literal["SN3", "SN5"]


# ---------------------------------------------------------------------------
# Lazy / guarded heavy-import helpers. Each raises a *clear* message telling the
# user exactly what to install, instead of a bare ImportError at module load.
# ---------------------------------------------------------------------------
def _require(module: str, pip_name: Optional[str] = None) -> Any:
    """Import ``module`` on demand or raise a friendly, actionable error."""
    pip_name = pip_name or module
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"'{module}' is required for this operation but is not installed. "
            f"Install it with:  pip install {pip_name}\n"
            "NOTE: on the 4 GB-GPU / Python-3.14 demo machine do NOT install "
            "torch/rasterio or download SpaceNet — this dataset is meant to run "
            "on a separate training box. See the module docstring."
        ) from exc


def _occlusion_module() -> Any:
    """Import the numpy-only occlusion generators from the FastAPI service.

    Imported lazily so the ML package does not depend on the backend app at
    import time. Falls back with a clear message if the path is not wired up.
    """
    try:
        from backend.app.services import occlusion  # type: ignore
        return occlusion
    except ImportError:
        try:
            from app.services import occlusion  # type: ignore
            return occlusion
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Could not import occlusion generators. Ensure the project root "
                "(containing 'backend/') is on PYTHONPATH, or copy "
                "backend/app/services/occlusion.py next to this module."
            ) from exc


# ---------------------------------------------------------------------------
# Config + a sample (image, geojson) pairing helper.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SpaceNetConfig:
    """Configuration for a SpaceNet centerline dataset split.

    Attributes:
        root:         Local directory of a *downloaded* AOI (NO S3 / no network).
        challenge:    "SN3" or "SN5" (SN5 adds per-segment speed -> travel time).
        image_subdir: Subfolder of GeoTIFF chips (e.g. "PS-RGB" or "PS-MS").
        label_subdir: Subfolder of ground-truth road geojsons ("geojson_roads").
        label_mode:   "buffer" (filled road mask) or "centerline" (thin ribbon).
        road_width_m: Default buffer half-width in metres for "buffer" mode when
                      a per-segment width/lane attribute is unavailable.
        chip_size:    Square crop size in pixels fed to the network.
        occlude:      If True, also produce a synthetically-occluded input view
                      (clean target preserved) for robustness / consistency loss.
        occlude_seed: Base RNG seed for reproducible occlusion (per-index offset).
    """

    root: str
    challenge: SpaceNetChallenge = "SN3"
    image_subdir: str = "PS-RGB"
    label_subdir: str = "geojson_roads"
    label_mode: LabelMode = "buffer"
    road_width_m: float = 2.0
    chip_size: int = 512
    occlude: bool = False
    occlude_seed: int = 1234
    image_glob: str = "*.tif"
    extra: dict[str, Any] = field(default_factory=dict)


def _stem_to_label_path(image_path: Path, cfg: SpaceNetConfig) -> Optional[Path]:
    """Map a SpaceNet image filename to its sibling road geojson, if present.

    SpaceNet keeps a stable ``imgN`` token in both names, e.g.::
        ..._PS-RGB_img42.tif  <->  ..._geojson_roads_img42.geojson
    We swap the image subdir/token for the label ones and also fall back to a
    permissive ``*imgN*.geojson`` glob.
    """
    label_dir = image_path.parent.parent / cfg.label_subdir
    if not label_dir.is_dir():
        return None
    # Extract the trailing imgN token.
    stem = image_path.stem
    token = stem.split("_")[-1]  # e.g. "img42"
    direct = label_dir / stem.replace(cfg.image_subdir, cfg.label_subdir)
    direct = direct.with_suffix(".geojson")
    if direct.is_file():
        return direct
    matches = sorted(label_dir.glob(f"*{token}*.geojson"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Rasterization helpers (vector centerline -> training mask). Guarded imports.
# ---------------------------------------------------------------------------
def load_centerlines(geojson_path: str | os.PathLike[str]) -> list[list[tuple[float, float]]]:
    """Read a SpaceNet road geojson into a list of LineString coordinate lists.

    Pure-Python (json only) — no shapely needed just to read coordinates. Each
    returned item is a list of ``(x, y)`` pixel-or-geo coordinate tuples. Note:
    SpaceNet geojsons are typically in *geographic* coords; converting to the
    chip's pixel grid requires the GeoTIFF affine transform (rasterio) — wired
    up in :func:`rasterize_label` / TODO for full georeferencing.
    """
    path = Path(geojson_path)
    with path.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)
    lines: list[list[tuple[float, float]]] = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "LineString":
            lines.append([(float(x), float(y)) for x, y, *_ in coords])
        elif gtype == "MultiLineString":
            for part in coords:
                lines.append([(float(x), float(y)) for x, y, *_ in part])
    return lines


def rasterize_label(geojson_path: str | os.PathLike[str],
                    image_path: str | os.PathLike[str],
                    cfg: SpaceNetConfig) -> "np.ndarray":
    """Rasterize road centerlines onto the chip's pixel grid -> float32 mask.

    Requires ``rasterio`` (for the GeoTIFF affine transform + shape) and
    ``shapely`` (to buffer/segment LineStrings). Raises a friendly ImportError
    listing the pip command if they are absent — so this stays import-safe and
    is only "expensive" when actually called on the training box.

    TODO(training-box): implement the geo->pixel transform and burn-in:
        * read transform + (H, W) from the GeoTIFF via rasterio,
        * reproject geojson coords to the image CRS if needed,
        * for "buffer": shapely buffer by (road_width_m or lane-derived width),
        * for "centerline": rasterize a 1–2 px ribbon (rasterio.features.rasterize),
        * return a float32 (H, W) array in [0, 1].
    """
    if np is None:  # pragma: no cover
        _require("numpy")
    rasterio = _require("rasterio")  # noqa: F841 - used in the TODO body
    _require("shapely")
    _ = (geojson_path, image_path, cfg)
    raise NotImplementedError(
        "rasterize_label is a scaffold. Implement the geo->pixel burn-in on the "
        "training box (see TODO in the docstring). Centerlines can be read "
        "library-free via load_centerlines()."
    )


# ---------------------------------------------------------------------------
# Dataset skeleton. Subclasses torch.utils.data.Dataset *only if torch is
# present*; otherwise it transparently uses ``object`` as the base so the file
# imports without torch. Iteration/instantiation that needs torch raises a
# clear message.
# ---------------------------------------------------------------------------
class SpaceNetCenterlineDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """PyTorch ``Dataset`` skeleton over a *local* SpaceNet SN3/SN5 AOI.

    Yields ``(image, target)`` (and, when ``cfg.occlude`` is True, an extra
    occluded-image view) for a road-segmentation network. Does NOT touch S3 and
    does NOT download anything — point ``cfg.root`` at a pre-downloaded AOI.

    Usage (on a training box, NOT the demo laptop)::

        cfg = SpaceNetConfig(root="/scratch/spacenet/SN3/AOI_2_Vegas",
                             challenge="SN3", label_mode="centerline",
                             occlude=True)
        ds = SpaceNetCenterlineDataset(cfg)
        img, tgt = ds[0]                 # clean
        # with occlude=True -> ds[i] == {"image", "image_occ", "target"}

    Wire into a DataLoader for training; report BOTH IoU/Dice (pixel) and
    APLS/connectivity (graph) on a held-out split.
    """

    def __init__(self, cfg: SpaceNetConfig,
                 transform: Optional[Callable[..., Any]] = None) -> None:
        self.cfg = cfg
        self.transform = transform
        root = Path(cfg.root)
        if not root.exists():
            # Don't hard-fail at construction in a scaffold; warn loudly so the
            # user knows to download an AOI locally first.
            print(
                f"[spacenet] WARNING: root '{root}' does not exist. Download one "
                "AOI locally first (see module docstring 'DATA ACCESS'). The "
                "dataset will be empty until then."
            )
            self.samples: list[tuple[Path, Optional[Path]]] = []
            return
        image_dir = root / cfg.image_subdir
        images = sorted(image_dir.glob(cfg.image_glob)) if image_dir.is_dir() else []
        self.samples = [(img, _stem_to_label_path(img, cfg)) for img in images]
        n_labeled = sum(1 for _, lbl in self.samples if lbl is not None)
        print(
            f"[spacenet] {cfg.challenge} '{root.name}': {len(self.samples)} chips "
            f"({n_labeled} with labels), label_mode={cfg.label_mode}, "
            f"occlude={cfg.occlude}."
        )

    # ---- standard Dataset protocol -------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, image_path: Path) -> "np.ndarray":
        """Read a GeoTIFF chip into an (H, W, C) uint8/float array (rasterio)."""
        rasterio = _require("rasterio")
        with rasterio.open(image_path) as src:  # type: ignore[attr-defined]
            arr = src.read()  # (C, H, W)
        if np is None:  # pragma: no cover
            _require("numpy")
        return np.transpose(arr, (1, 2, 0))  # -> (H, W, C)

    def __getitem__(self, index: int) -> Any:
        if np is None:  # pragma: no cover
            _require("numpy")
        image_path, label_path = self.samples[index]
        image = self._read_image(image_path)
        target = (
            rasterize_label(label_path, image_path, self.cfg)
            if label_path is not None
            else np.zeros(image.shape[:2], dtype=np.float32)
        )

        sample: dict[str, Any] = {"image": image, "target": target}
        if self.cfg.occlude:
            occlusion = _occlusion_module()
            seed = self.cfg.occlude_seed + index
            sample["image_occ"] = occlusion.random_occlude(image, seed=seed)

        if self.transform is not None:
            sample = self.transform(sample)

        # Convenience: return a plain (image, target) tuple unless occlusion is
        # on (then the consistency-loss training loop wants the full dict).
        if not self.cfg.occlude and self.transform is None:
            return sample["image"], sample["target"]
        return sample


# ---------------------------------------------------------------------------
# Runnable smoke test — import-only checks, NO download, NO torch required.
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Verify the module imports and the empty-root path behaves, offline."""
    cfg = SpaceNetConfig(
        root=os.environ.get("SPACENET_ROOT", "/tmp/__nonexistent_spacenet__"),
        challenge="SN3",
        label_mode="centerline",
        occlude=True,
    )
    ds = SpaceNetCenterlineDataset(cfg)
    print(f"[spacenet] smoke: dataset length = {len(ds)} (0 if root missing).")
    print("[spacenet] smoke: module imported OK without torch/rasterio.")
    print(
        "[spacenet] TODO: set SPACENET_ROOT to a locally-downloaded AOI and "
        "install torch+rasterio+shapely on a training box to exercise __getitem__."
    )


if __name__ == "__main__":
    _smoke_test()
