"""Albumentations-compatible synthetic occlusion transforms (training side).

These wrap the numpy-only occlusion generators from
``backend/app/services/occlusion.py`` (clouds / shadows / tree-canopy) and add a
fourth **building** occluder, exposing each as a custom ``albumentations``
transform so they slot directly into a training pipeline:

    A.Compose([CloudOcclusion(p=0.6), ShadowOcclusion(p=0.7),
               CanopyOcclusion(p=0.7), BuildingOcclusion(p=0.3), ...])

The transforms occlude the INPUT image only and leave the road mask untouched —
the network must still predict the true road under the occlusion. That is the
robustness INNOVATION: pair a clean view with an occluded view of the same chip
and add a consistency loss (``ml/train/losses.py``) so the model learns to "see
through" clouds, shadows, canopy and building shadows. This is what lets the
*robust* model recover the critical links the *baseline* loses under occlusion
(``input=occluded & model=baseline`` is the deliberately fragmented demo case).

Design notes
------------
* The pixel math is numpy-only and re-implements the SAME soft-blob recipe as
  the backend service (``_blob_mask`` + per-occluder tints/darkening) so the
  synthetic occlusion seen at train time matches what the API describes.
* ``albumentations`` is imported lazily via ``_require_albumentations`` so this
  file is import-safe on machines without it (linting / CI / the API image,
  which has no heavy ML deps). The raw numpy ``apply_*`` functions below need
  ONLY numpy and can be used standalone without albumentations installed.
* The albumentations transform classes are built dynamically inside
  ``_build_transforms`` (after the lazy import) and re-exported at module level
  on first access via ``__getattr__`` — so ``from ml.aug.occlusion import
  CloudOcclusion`` works when albumentations is present and fails with a clear,
  actionable message when it is not, without breaking ``import ml.aug.occlusion``.

Pinned deps (see ml/requirements-ml.txt — do NOT pip install here):
    albumentations==1.4.18
    numpy==2.1.3
    # opencv-python(-headless)==4.10.0.84   # albumentations backend

Usage (TODO: wire into ml/train/train.py)
-----------------------------------------
    # >>> import albumentations as A
    # >>> from ml.aug.occlusion import build_occlusion_compose
    # >>> occ = build_occlusion_compose(p_any=0.9)     # albumentations Compose
    # >>> sample = occ(image=chip_uint8)               # mask is NOT occluded
    # >>> occluded = sample["image"]

Or standalone (numpy only, no albumentations needed):
    # >>> from ml.aug.occlusion import random_occlude
    # >>> occluded = random_occlude(chip_uint8, seed=0)

Run ``python -m ml.aug.occlusion`` for a tiny self-test on a synthetic image
(works numpy-only; reports gracefully if albumentations is absent).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # only for type hints; never imported at runtime
    import albumentations as A


# ---------------------------------------------------------------------------
# numpy-only occlusion math (mirrors backend/app/services/occlusion.py)
# ---------------------------------------------------------------------------
def _blob_mask(h: int, w: int, n: int, rng: np.random.Generator,
               r_min: float, r_max: float) -> np.ndarray:
    """Soft 0..1 occlusion mask = union of ``n`` Gaussian blobs.

    Identical recipe to the backend service so train-time occlusion matches the
    occlusion the API story describes.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        rad = rng.uniform(r_min, r_max) * min(h, w)
        soft = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        mask = np.maximum(mask, soft)
    return np.clip(mask, 0, 1)


def apply_clouds(img: np.ndarray, intensity: float = 0.9,
                 seed: int | None = None) -> np.ndarray:
    """Bright, large soft blobs blended toward white — cloud cover."""
    rng = np.random.default_rng(seed)
    m = _blob_mask(*img.shape[:2], n=int(rng.integers(1, 4)), rng=rng,
                   r_min=0.08, r_max=0.22)[..., None]
    return (img * (1 - m * intensity) + 255 * m * intensity).astype(img.dtype)


def apply_shadows(img: np.ndarray, darkness: float = 0.55,
                  seed: int | None = None) -> np.ndarray:
    """Darkening soft blobs — cast shadows."""
    rng = np.random.default_rng(seed)
    m = _blob_mask(*img.shape[:2], n=int(rng.integers(2, 6)), rng=rng,
                   r_min=0.04, r_max=0.12)[..., None]
    return (img * (1 - m * darkness)).astype(img.dtype)


def apply_canopy(img: np.ndarray, strength: float = 0.6,
                 seed: int | None = None) -> np.ndarray:
    """Green-tinted irregular patches mimicking tree-cover occlusion."""
    rng = np.random.default_rng(seed)
    m = _blob_mask(*img.shape[:2], n=int(rng.integers(3, 8)), rng=rng,
                   r_min=0.03, r_max=0.09)[..., None]
    green = np.zeros_like(img)
    green[..., 1] = 110
    return (img * (1 - m * strength) + green * m * strength).astype(img.dtype)


def apply_building(img: np.ndarray, strength: float = 0.85,
                   seed: int | None = None) -> np.ndarray:
    """Hard-edged rectangular blocks (+ a darker offset shadow) — buildings/
    tall-structure occlusion that hides roads in dense urban tiles.

    Unlike the soft-blob occluders this uses sharp rectangles: buildings have
    crisp footprints, and their shadows fall to one side. Roughly the same
    visual contract as the backend service's soft occluders, extended for the
    urban-mobility (Bengaluru) demo.
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32).copy()
    grey = np.array([70.0, 72.0, 78.0], dtype=np.float32)   # concrete-ish block
    shadow_dark = 0.5                                       # cast-shadow factor
    n = int(rng.integers(2, 6))
    for _ in range(n):
        bh = int(rng.uniform(0.06, 0.20) * h)
        bw = int(rng.uniform(0.06, 0.20) * w)
        y0 = int(rng.uniform(0, max(1, h - bh)))
        x0 = int(rng.uniform(0, max(1, w - bw)))
        # cast shadow: a same-size block offset down-right, darkening the ground
        sy = min(h, y0 + bh + int(0.4 * bh))
        sx = min(w, x0 + bw + int(0.4 * bw))
        s_y0, s_x0 = min(y0 + int(0.4 * bh), h), min(x0 + int(0.4 * bw), w)
        out[s_y0:sy, s_x0:sx] *= (1 - shadow_dark)
        # the building block itself: blend toward the flat grey
        out[y0:y0 + bh, x0:x0 + bw] = (
            out[y0:y0 + bh, x0:x0 + bw] * (1 - strength) + grey * strength
        )
    return np.clip(out, 0, 255).astype(img.dtype)


def random_occlude(img: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Stochastically stack the four occluders — numpy-only, no albumentations.

    Mirrors ``backend/app/services/occlusion.py:random_occlude`` and adds the
    building occluder. Useful for quick offline previews and unit tests.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    if rng.random() < 0.6:
        out = apply_clouds(out, seed=seed)
    if rng.random() < 0.7:
        out = apply_shadows(out, seed=seed)
    if rng.random() < 0.7:
        out = apply_canopy(out, seed=seed)
    if rng.random() < 0.3:
        out = apply_building(out, seed=seed)
    return out


# Maps a friendly occluder name -> its numpy apply function (no albu needed).
OCCLUDERS: dict[str, Callable[..., np.ndarray]] = {
    "cloud": apply_clouds,
    "shadow": apply_shadows,
    "canopy": apply_canopy,
    "building": apply_building,
}


# ---------------------------------------------------------------------------
# albumentations integration (lazy — file imports fine without albumentations)
# ---------------------------------------------------------------------------
def _require_albumentations():
    """Import albumentations lazily so the module is import-safe without it.

    Returns the ``albumentations`` module, or raises a clear, actionable error
    (instead of a bare ``ModuleNotFoundError`` deep in a stack trace).
    """
    try:
        import albumentations as A  # noqa: F401  (returned to caller)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "albumentations is required for the occlusion transform classes. "
            "Install the pinned version (do NOT pip install in this hackathon "
            "env): `pip install albumentations==1.4.18`. The numpy-only "
            "`apply_*` / `random_occlude` functions work without it."
        ) from exc
    return A


# Cache for the dynamically-built transform classes (built once, after import).
_TRANSFORM_CACHE: dict[str, type] | None = None


def _build_transforms() -> dict[str, type]:
    """Build the albumentations transform classes after a successful lazy import.

    Each class subclasses ``albumentations.ImageOnlyTransform`` so it is applied
    to the image only and the road mask is preserved — exactly what the
    clean-vs-occluded consistency objective needs.
    """
    global _TRANSFORM_CACHE
    if _TRANSFORM_CACHE is not None:
        return _TRANSFORM_CACHE
    A = _require_albumentations()

    class _OcclusionTransform(A.ImageOnlyTransform):
        """Base: call a numpy ``apply_*`` occluder on the image only."""

        #: subclasses set this to one of the OCCLUDERS functions
        _fn: Callable[..., np.ndarray]

        def __init__(self, p: float = 0.5,
                     always_apply: bool | None = None, **fn_kwargs: Any) -> None:
            # albumentations 1.4.x still accepts always_apply; keep it optional
            # so the same call works across minor versions.
            if always_apply is None:
                super().__init__(p=p)
            else:  # pragma: no cover - version-dependent branch
                super().__init__(p=p, always_apply=always_apply)
            self._fn_kwargs = fn_kwargs

        def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
            # albumentations draws its own seed per-call; let numpy default.
            return type(self)._fn(img, **self._fn_kwargs)

        def get_transform_init_args_names(self) -> tuple[str, ...]:
            return tuple(self._fn_kwargs.keys())

    class CloudOcclusion(_OcclusionTransform):
        """Synthetic cloud cover (bright soft blobs)."""
        _fn = staticmethod(apply_clouds)

    class ShadowOcclusion(_OcclusionTransform):
        """Synthetic cast shadows (darkening soft blobs)."""
        _fn = staticmethod(apply_shadows)

    class CanopyOcclusion(_OcclusionTransform):
        """Synthetic tree-canopy cover (green-tinted patches)."""
        _fn = staticmethod(apply_canopy)

    class BuildingOcclusion(_OcclusionTransform):
        """Synthetic building/tall-structure occlusion (hard blocks + shadow)."""
        _fn = staticmethod(apply_building)

    _TRANSFORM_CACHE = {
        "CloudOcclusion": CloudOcclusion,
        "ShadowOcclusion": ShadowOcclusion,
        "CanopyOcclusion": CanopyOcclusion,
        "BuildingOcclusion": BuildingOcclusion,
    }
    return _TRANSFORM_CACHE


def build_occlusion_compose(p_any: float = 0.9,
                            p_cloud: float = 0.6, p_shadow: float = 0.7,
                            p_canopy: float = 0.7, p_building: float = 0.3) -> "A.Compose":
    """Return an ``albumentations.Compose`` of the four occlusion transforms.

    Wrapped in ``A.OneOf``-free sequential ``A.SomeOf`` semantics: each occluder
    fires independently with its own probability (so multiple can stack, matching
    ``random_occlude``), and the whole group fires with probability ``p_any``.

    Requires albumentations (raises a clear error otherwise).
    """
    A = _require_albumentations()
    t = _build_transforms()
    return A.Compose(
        [
            t["CloudOcclusion"](p=p_cloud),
            t["ShadowOcclusion"](p=p_shadow),
            t["CanopyOcclusion"](p=p_canopy),
            t["BuildingOcclusion"](p=p_building),
        ],
        p=p_any,
    )


# Names resolved lazily through __getattr__ so importing the module never needs
# albumentations, but `from ml.aug.occlusion import CloudOcclusion` still works.
_LAZY_TRANSFORM_NAMES = (
    "CloudOcclusion", "ShadowOcclusion", "CanopyOcclusion", "BuildingOcclusion",
)


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attribute
    if name in _LAZY_TRANSFORM_NAMES:
        return _build_transforms()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_TRANSFORM_NAMES))


# ---------------------------------------------------------------------------
# self-test / usage demo
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Run each occluder on a synthetic image; verify shape/dtype are preserved.

    numpy-only path always runs; the albumentations path runs only if installed.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    for name, fn in OCCLUDERS.items():
        out = fn(img, seed=0)
        assert out.shape == img.shape, f"{name}: shape changed"
        assert out.dtype == img.dtype, f"{name}: dtype changed"
        changed = float(np.mean(np.abs(out.astype(np.int32) - img.astype(np.int32))))
        print(f"  [numpy ] {name:<9} ok  mean|delta|={changed:6.2f}")

    combined = random_occlude(img, seed=0)
    assert combined.shape == img.shape and combined.dtype == img.dtype
    print("  [numpy ] random_occlude ok")

    try:
        compose = build_occlusion_compose(p_any=1.0)
    except ImportError as exc:
        print(f"  [albu  ] skipped — {exc}")
        return
    sample = compose(image=img)
    out = sample["image"]
    assert out.shape == img.shape and out.dtype == img.dtype
    print("  [albu  ] build_occlusion_compose ok")


if __name__ == "__main__":
    print("ml/aug/occlusion.py self-test")
    _self_test()
    print("done.")
