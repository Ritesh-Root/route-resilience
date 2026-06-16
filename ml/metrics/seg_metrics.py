"""Binary segmentation metrics for occlusion-robust road extraction (numpy-only).

Implements the three core mask metrics used by the backend `/api/metrics`
endpoint (see BACKEND CONTRACT: {iou, dice, occlusionRecall, ...}):

  * IoU (Jaccard)         — intersection-over-union of predicted vs. true roads.
  * Dice (F1 over pixels) — 2|P∩T| / (|P|+|T|).
  * Occlusion-Recall      — recall computed *only* over pixels that were hidden
                            by synthetic occlusion. This is the headline number
                            for the robustness story: how many road pixels the
                            model still recovers *underneath* clouds / shadows /
                            canopy. The occlusion mask is exactly the kind of
                            soft blob produced by
                            `backend/app/services/occlusion.py`; threshold it to
                            booleans before passing it here.

Pure numpy so this stays importable on the constrained box (Python 3.14, 4GB
GPU) with no torch / rasterio required. Heavy/optional imports are guarded.

Usage (no model needed):

    python -m ml.metrics.seg_metrics      # runs the self-test in __main__

    >>> from ml.metrics.seg_metrics import compute_seg_metrics
    >>> m = compute_seg_metrics(pred, target, occlusion_mask=occ)
    >>> m["iou"], m["dice"], m["occlusionRecall"]
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["iou", "dice", "occlusion_recall", "compute_seg_metrics"]

# Smoothing term to keep metrics defined when a mask is entirely empty.
_EPS: float = 1e-7


def _as_bool(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Coerce an arbitrary mask array to a boolean foreground array.

    Accepts bool, integer label, or float/probability masks. Float masks are
    thresholded at ``threshold`` (matches how soft occlusion blobs from
    occlusion.py are binarised). NaNs are treated as background.
    """
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return np.nan_to_num(arr, nan=0.0) >= threshold
    # Integer / other: any non-zero is foreground.
    return arr != 0


def _check_shape(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share a shape, got {pred.shape} vs {target.shape}"
        )


def iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Intersection-over-Union (Jaccard index) for the foreground class.

    Returns 1.0 when both masks are empty (perfect agreement on "no road").
    """
    p = _as_bool(pred, threshold)
    t = _as_bool(target, threshold)
    _check_shape(p, t)
    intersection = float(np.logical_and(p, t).sum())
    union = float(np.logical_or(p, t).sum())
    if union == 0.0:
        return 1.0
    return intersection / (union + _EPS)


def dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Dice coefficient (== F1 over pixels) for the foreground class.

    Returns 1.0 when both masks are empty.
    """
    p = _as_bool(pred, threshold)
    t = _as_bool(target, threshold)
    _check_shape(p, t)
    intersection = float(np.logical_and(p, t).sum())
    total = float(p.sum() + t.sum())
    if total == 0.0:
        return 1.0
    return (2.0 * intersection) / (total + _EPS)


def occlusion_recall(
    pred: np.ndarray,
    target: np.ndarray,
    occlusion_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Recall computed *only* over occluded pixels.

    Of the true road pixels that fell under occlusion (clouds / shadow / canopy),
    what fraction did the model still predict as road? This isolates the
    robustness gain: the baseline model collapses here under occlusion while the
    robust model recovers the hidden links.

    ``occlusion_mask`` marks which pixels were occluded in the input (the soft
    blob from occlusion.py, thresholded). It is intersected with the true road
    mask, so only occluded *road* pixels count toward the denominator.

    Returns 1.0 when there are no occluded road pixels (nothing to recover).
    """
    p = _as_bool(pred, threshold)
    t = _as_bool(target, threshold)
    occ = _as_bool(occlusion_mask, threshold)
    _check_shape(p, t)
    _check_shape(t, occ)

    occluded_road = np.logical_and(t, occ)
    denom = float(occluded_road.sum())
    if denom == 0.0:
        return 1.0
    recovered = float(np.logical_and(p, occluded_road).sum())
    return recovered / (denom + _EPS)


def compute_seg_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    occlusion_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all binary-mask metrics in one pass.

    Returns a dict keyed to match the backend `/api/metrics` contract:
    ``{"iou", "dice", "occlusionRecall"}``. ``occlusionRecall`` is ``None`` when
    no occlusion mask is supplied (e.g. the ``input=clean`` case).

    TODO(api): graph metrics (connectivityRatio, apls, resilienceIndex) are
    computed on the road *graph*, not the pixel mask — see the graph services in
    backend/app/services. This module owns only the pixel-level numbers.
    """
    result: dict[str, float] = {
        "iou": iou(pred, target, threshold),
        "dice": dice(pred, target, threshold),
    }
    result["occlusionRecall"] = (
        occlusion_recall(pred, target, occlusion_mask, threshold)
        if occlusion_mask is not None
        else None  # type: ignore[assignment]
    )
    return result


if __name__ == "__main__":
    # Self-test: no model, no dataset, no GPU. Deterministic synthetic masks
    # exercise the perfect, partial, and empty cases.
    rng = np.random.default_rng(0)

    # Ground-truth "road": a couple of straight lines on a 64x64 grid.
    truth = np.zeros((64, 64), dtype=bool)
    truth[30, :] = True          # horizontal road
    truth[:, 20] = True          # vertical road

    # Occlusion blob covering the right half (mimics a cloud over part of a road).
    occ = np.zeros((64, 64), dtype=bool)
    occ[:, 40:] = True

    # Baseline-style prediction: recovers the un-occluded road but loses the
    # horizontal road where the cloud sits (columns >= 40).
    baseline = truth.copy()
    baseline[30, 40:] = False

    # Robust-style prediction: recovers everything, with a little extra noise.
    robust = truth.copy()
    robust[10, 5] = True

    print("== baseline (loses occluded road) ==")
    print(compute_seg_metrics(baseline, truth, occ))
    print("== robust (recovers occluded road) ==")
    print(compute_seg_metrics(robust, truth, occ))
    print("== both-empty edge case (should be 1.0 / 1.0 / 1.0) ==")
    empty = np.zeros((8, 8), dtype=bool)
    print(compute_seg_metrics(empty, empty, empty))

    # Sanity assertions so the self-test fails loudly if a metric regresses.
    bm = compute_seg_metrics(baseline, truth, occ)
    assert bm["occlusionRecall"] < 1.0, "baseline should miss occluded road"
    rm = compute_seg_metrics(robust, truth, occ)
    assert rm["occlusionRecall"] > 0.999, "robust should recover occluded road"
    print("\nself-test OK")
