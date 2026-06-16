"""Road mask -> 1px centerline skeleton (VECTORIZE side, step 1 of mask->graph).

The segmentation model emits a per-pixel road *probability* map. Before we can
build the resilience graph (nodes = junctions, edges = road segments with
``travelTimeSec`` / ``lengthM`` / ``criticality``), that soft mask must be
reduced to a clean **1-pixel-wide centerline skeleton**. This module is exactly
that bridge:

    probability map -> threshold -> morphological clean-up -> skeletonize

The skeleton produced here is the input to the *next* stage (skeleton -> graph:
trace branches into LineStrings, snap junctions to nodes), which lives elsewhere
in ``ml/vectorize/``. Centerlines (not filled footprints) are what SpaceNet's
APLS grader and our backend's edge model both want — see ``ml/data/spacenet.py``.

Two skeleton backends are offered, mirroring common road-extraction pipelines:

  * ``"zhang"``  — :func:`skimage.morphology.skeletonize` (Zhang-Suen / Lee).
    Fast, thin, topology-preserving; the usual default.
  * ``"medial"`` — :func:`skimage.morphology.medial_axis`, the **FilFinder-style
    medial axis**. FilFinder (an astronomy filament tracer) popularised driving
    a medial-axis skeleton from a distance transform; the returned distance map
    doubles as a per-pixel **road half-width** estimate, handy later for pruning
    spurs and for lane/width attributes. We expose that distance map too.

Design constraints (HARD): the demo box runs Python 3.14 on a 4 GB RTX 3050.
This file is **pure scikit-image / numpy** — no torch, no GDAL — and every heavy
import is guarded so ``import ml.vectorize.skeletonize`` ALWAYS succeeds even on a
bare interpreter (scikit-image absent). The libs are only needed when you call a
function. No dataset is downloaded and no model is run here.

The blob/soft-mask intuition here is the post-processing mirror of the *training*
side occlusion blobs in ``backend/app/services/occlusion.py`` (numpy-only there
too): occlusion punches holes in the input, so the predicted mask often has small
gaps/specks — the morphology step below (close gaps, drop specks) is what cleans
those artefacts before skeletonization.

------------------------------------------------------------------------------
Usage (no GeoTIFF / dataset needed — runs on a synthetic mask by default)::

    # self-contained demo on a generated synthetic road mask:
    python "ml/vectorize/skeletonize.py" --demo --save-prefix /tmp/skel_demo

    # real probability map saved as a .npy (H x W float in [0,1] or uint8):
    python "ml/vectorize/skeletonize.py" --mask pred_prob.npy \\
        --threshold 0.5 --method medial --save-prefix /tmp/out

TODO(you): feed ``SkeletonResult.skeleton`` into the skeleton->graph tracer
(branch/junction detection -> shapely LineStrings -> networkx graph) so the
backend gatekeeper / resilience logic can consume real extracted roads.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-checker only; never imported at runtime
    import numpy.typing as npt

# Skeletonization backends. "zhang" = Zhang-Suen/Lee thinning; "medial" =
# FilFinder-style medial axis (also yields a distance/half-width map).
SkeletonMethod = Literal["zhang", "medial"]


@dataclass
class SkeletonResult:
    """Outputs of the mask -> skeleton pipeline.

    Attributes:
        binary:   Cleaned boolean road mask after threshold + morphology.
        skeleton: Boolean 1-pixel-wide centerline skeleton.
        distance: Per-pixel distance-to-background (road *half-width* proxy),
            only populated for ``method="medial"``; ``None`` otherwise.
        method:   Which backend produced ``skeleton``.
        n_skel_px: Number of True pixels in ``skeleton`` (quick sanity metric).
    """

    binary: "npt.NDArray[np.bool_]"
    skeleton: "npt.NDArray[np.bool_]"
    distance: "npt.NDArray[np.float64] | None"
    method: SkeletonMethod
    n_skel_px: int


def _require_skimage():
    """Import scikit-image lazily with a clear, actionable error message.

    Guarded so this module imports on a bare Python 3.14 interpreter; the heavy
    dep is only needed when a pipeline function is actually called.
    """
    try:
        import skimage  # noqa: F401  (presence check)
    except Exception as exc:  # ImportError or C-ext load failure
        raise RuntimeError(
            "scikit-image is required for skeletonization but could not be "
            f"imported ({type(exc).__name__}: {exc}). Install it with "
            "`pip install scikit-image==0.24.0` (see ml/requirements-ml.txt). "
            "On Python 3.14, if no cp314 wheel exists, use a py3.12 conda env."
        ) from exc


def threshold_mask(
    prob: "npt.NDArray[np.floating | np.integer]",
    threshold: float = 0.5,
    *,
    use_otsu: bool = False,
) -> "npt.NDArray[np.bool_]":
    """Binarize a road probability/intensity map.

    Args:
        prob: 2-D array. Floats are treated as probabilities in [0, 1]; integer
            arrays (e.g. uint8 0..255) are normalized to [0, 1] by their max.
        threshold: Fixed cut applied when ``use_otsu`` is False.
        use_otsu: If True, ignore ``threshold`` and pick it via Otsu's method
            (robust when the prob map is not well calibrated around 0.5).

    Returns:
        Boolean foreground (road) mask.
    """
    arr = np.asarray(prob)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {arr.shape!r}")

    # Normalize integer maps (0..255) to floats in [0, 1]; leave floats as-is.
    if np.issubdtype(arr.dtype, np.integer):
        peak = float(arr.max())
        arr = arr.astype(np.float64) / peak if peak > 0 else arr.astype(np.float64)
    else:
        arr = arr.astype(np.float64)

    if use_otsu:
        _require_skimage()
        from skimage.filters import threshold_otsu

        # threshold_otsu fails on a single-valued image; fall back to >0 then.
        if np.isclose(arr.min(), arr.max()):
            return arr > arr.min()
        threshold = float(threshold_otsu(arr))

    return arr >= threshold


def clean_mask(
    binary: "npt.NDArray[np.bool_]",
    *,
    close_radius: int = 2,
    min_object_size: int = 64,
    fill_hole_size: int = 64,
) -> "npt.NDArray[np.bool_]":
    """Morphological clean-up before skeletonization.

    Occlusion (clouds/shadows/canopy) tends to leave gaps in the predicted road
    mask and scatter small false specks. We:
      1. **close** (dilate then erode) with a disk to bridge small gaps so a road
         interrupted by an occlusion blob reconnects into one component;
      2. **remove small objects** to drop isolated speckle that would otherwise
         become spurious skeleton fragments;
      3. **fill small holes** so a solid road body skeletonizes to its centerline
         rather than a ring around the hole.

    Args:
        binary: Boolean road mask.
        close_radius: Disk radius (px) for the morphological closing; 0 disables.
        min_object_size: Connected components smaller than this (px) are removed;
            0 disables.
        fill_hole_size: Holes smaller than this (px) are filled; 0 disables.

    Returns:
        Cleaned boolean mask, same shape.
    """
    _require_skimage()
    from skimage.morphology import (
        binary_closing,
        disk,
        remove_small_holes,
        remove_small_objects,
    )

    out = np.asarray(binary, dtype=bool)
    if close_radius > 0:
        out = binary_closing(out, disk(close_radius))
    if min_object_size > 0:
        out = remove_small_objects(out, min_size=min_object_size)
    if fill_hole_size > 0:
        out = remove_small_holes(out, area_threshold=fill_hole_size)
    return out


def skeletonize_mask(
    binary: "npt.NDArray[np.bool_]",
    method: SkeletonMethod = "zhang",
) -> tuple["npt.NDArray[np.bool_]", "npt.NDArray[np.float64] | None"]:
    """Reduce a binary road mask to a 1-pixel-wide centerline skeleton.

    Args:
        binary: Cleaned boolean road mask.
        method: ``"zhang"`` for Zhang-Suen/Lee thinning (skeletonize), or
            ``"medial"`` for the FilFinder-style medial axis.

    Returns:
        ``(skeleton, distance)`` where ``skeleton`` is a boolean 1-px skeleton
        and ``distance`` is the medial-axis distance map (road half-width proxy)
        for ``"medial"``, else ``None``.
    """
    _require_skimage()
    arr = np.asarray(binary, dtype=bool)

    if method == "zhang":
        from skimage.morphology import skeletonize

        return skeletonize(arr), None
    if method == "medial":
        from skimage.morphology import medial_axis

        # return_distance gives the distance transform on the skeleton ridge —
        # the FilFinder trick for a per-pixel width estimate.
        skel, dist = medial_axis(arr, return_distance=True)
        return skel, np.asarray(dist, dtype=np.float64)

    raise ValueError(f"unknown method {method!r}; use 'zhang' or 'medial'")


def mask_to_skeleton(
    prob: "npt.NDArray[np.floating | np.integer]",
    *,
    threshold: float = 0.5,
    use_otsu: bool = False,
    method: SkeletonMethod = "zhang",
    close_radius: int = 2,
    min_object_size: int = 64,
    fill_hole_size: int = 64,
) -> SkeletonResult:
    """End-to-end: probability map -> threshold -> morphology -> skeleton.

    This is the single entry point most callers want. See the module docstring
    for the pipeline rationale and the per-step keyword arguments above.

    Args:
        prob: 2-D road probability/intensity map (float in [0,1] or integer).
        threshold, use_otsu: Passed to :func:`threshold_mask`.
        method: Skeleton backend, see :func:`skeletonize_mask`.
        close_radius, min_object_size, fill_hole_size: Passed to
            :func:`clean_mask`.

    Returns:
        A :class:`SkeletonResult` bundling the cleaned mask, skeleton, and
        (for the medial axis) the distance/half-width map.
    """
    binary = threshold_mask(prob, threshold=threshold, use_otsu=use_otsu)
    binary = clean_mask(
        binary,
        close_radius=close_radius,
        min_object_size=min_object_size,
        fill_hole_size=fill_hole_size,
    )
    skeleton, distance = skeletonize_mask(binary, method=method)
    return SkeletonResult(
        binary=binary,
        skeleton=skeleton,
        distance=distance,
        method=method,
        n_skel_px=int(skeleton.sum()),
    )


def _synthetic_road_mask(
    size: int = 256,
    *,
    occlude: bool = True,
    seed: int | None = 0,
) -> "npt.NDArray[np.float64]":
    """Generate a fake road probability map for the ``--demo`` mode (numpy-only).

    Draws a few thick straight "roads" on a blank canvas, then (optionally) punches
    a soft circular hole to imitate an occlusion gap — the same failure mode the
    morphology step is meant to repair. No skimage needed so ``--demo`` works even
    before scikit-image is installed (skeletonization still requires it).
    """
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    prob = np.zeros((h, w), dtype=np.float64)
    half = 3  # road half-thickness in px

    # A small grid of horizontal + vertical roads.
    for r in (size // 4, size // 2, 3 * size // 4):
        prob[np.abs(yy - r) <= half] = 1.0
    for c in (size // 4, size // 2, 3 * size // 4):
        prob[np.abs(xx - c) <= half] = 1.0

    if occlude:
        # Soft occlusion blob (cf. occlusion._blob_mask) that erases part of a road.
        cy, cx = size // 2, int(size * 0.62)
        rad = size * 0.06
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        prob = np.clip(prob - blob, 0.0, 1.0)

    # A little noise so a fixed 0.5 threshold is meaningful.
    prob = np.clip(prob + rng.normal(0, 0.02, prob.shape), 0.0, 1.0)
    return prob


def _save_png(arr: "npt.NDArray", path: str) -> bool:
    """Best-effort PNG dump for visual inspection; returns False if it can't."""
    try:
        from skimage.io import imsave
    except Exception:
        return False
    a = np.asarray(arr)
    if a.dtype == bool:
        a = (a * 255).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.floating):
        peak = float(a.max())
        a = (a / peak * 255).astype(np.uint8) if peak > 0 else a.astype(np.uint8)
    imsave(path, a, check_contrast=False)
    return True


def _load_mask(path: str) -> "npt.NDArray":
    """Load a probability map from .npy (numpy-only) or an image file (skimage)."""
    if path.lower().endswith(".npy"):
        return np.load(path)
    _require_skimage()
    from skimage.io import imread

    arr = imread(path)
    if arr.ndim == 3:  # collapse RGB(A) to a single channel
        arr = arr[..., :3].mean(axis=-1)
    return arr


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Road mask -> threshold -> morphology -> skeleton "
        "(pure scikit-image / numpy).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mask", help="path to a 2-D mask (.npy, or an image file)")
    src.add_argument("--demo", action="store_true",
                     help="run on a generated synthetic occluded road mask")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="binarization threshold (ignored with --otsu)")
    p.add_argument("--otsu", action="store_true",
                   help="pick the threshold via Otsu's method")
    p.add_argument("--method", choices=("zhang", "medial"), default="zhang",
                   help="skeleton backend (medial = FilFinder-style medial axis)")
    p.add_argument("--close-radius", type=int, default=2,
                   help="disk radius for morphological closing (0 disables)")
    p.add_argument("--min-object-size", type=int, default=64,
                   help="drop connected components smaller than this (0 disables)")
    p.add_argument("--fill-hole-size", type=int, default=64,
                   help="fill holes smaller than this (0 disables)")
    p.add_argument("--save-prefix", default=None,
                   help="if set, write <prefix>_binary.png / _skeleton.png "
                        "(+ _distance.png for medial) for inspection")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.demo:
        print("[demo] generating a 256x256 synthetic occluded road mask")
        prob = _synthetic_road_mask(256, occlude=True, seed=0)
    else:
        print(f"[load] {args.mask}")
        prob = _load_mask(args.mask)

    result = mask_to_skeleton(
        prob,
        threshold=args.threshold,
        use_otsu=args.otsu,
        method=args.method,
        close_radius=args.close_radius,
        min_object_size=args.min_object_size,
        fill_hole_size=args.fill_hole_size,
    )

    print(
        f"[done] method={result.method} "
        f"binary_px={int(result.binary.sum())} "
        f"skeleton_px={result.n_skel_px}"
    )
    if result.distance is not None:
        print(f"       medial half-width: max={result.distance.max():.2f}px")

    if args.save_prefix:
        wrote_any = _save_png(result.binary, f"{args.save_prefix}_binary.png")
        wrote_any |= _save_png(result.skeleton, f"{args.save_prefix}_skeleton.png")
        if result.distance is not None:
            wrote_any |= _save_png(result.distance, f"{args.save_prefix}_distance.png")
        if wrote_any:
            print(f"[save] wrote PNGs with prefix {args.save_prefix!r}")
        else:
            print("[save] skimage.io unavailable -> skipped PNG export")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
