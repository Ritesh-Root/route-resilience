"""Synthetic occlusion generators — the robustness INNOVATION (training side).

These run during MODEL TRAINING (apply to input chips, keep the true mask as
target) so the network learns to "see through" clouds, shadows and tree canopy.
For the strongest result, also forward a clean + an occluded copy of each chip
and add a consistency loss between the two predictions (see train/ in the plan).

numpy-only so the API image has no heavy CV deps. Plug into albumentations as a
custom transform during training.
"""
from __future__ import annotations

import numpy as np


def _blob_mask(h: int, w: int, n: int, rng: np.random.Generator,
               r_min: float, r_max: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        rad = rng.uniform(r_min, r_max) * min(h, w)
        soft = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        mask = np.maximum(mask, soft)
    return np.clip(mask, 0, 1)


def add_clouds(img: np.ndarray, intensity: float = 0.9, seed: int | None = None):
    rng = np.random.default_rng(seed)
    m = _blob_mask(*img.shape[:2], n=rng.integers(1, 4), rng=rng,
                   r_min=0.08, r_max=0.22)[..., None]
    return (img * (1 - m * intensity) + 255 * m * intensity).astype(img.dtype)


def add_shadows(img: np.ndarray, darkness: float = 0.55, seed: int | None = None):
    rng = np.random.default_rng(seed)
    m = _blob_mask(*img.shape[:2], n=rng.integers(2, 6), rng=rng,
                   r_min=0.04, r_max=0.12)[..., None]
    return (img * (1 - m * darkness)).astype(img.dtype)


def add_canopy(img: np.ndarray, strength: float = 0.6, seed: int | None = None):
    """Green-tinted irregular patches mimicking tree-cover occlusion."""
    rng = np.random.default_rng(seed)
    m = _blob_mask(*img.shape[:2], n=rng.integers(3, 8), rng=rng,
                   r_min=0.03, r_max=0.09)[..., None]
    green = np.zeros_like(img); green[..., 1] = 110
    return (img * (1 - m * strength) + green * m * strength).astype(img.dtype)


def random_occlude(img: np.ndarray, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    if rng.random() < 0.6:
        out = add_clouds(out, seed=seed)
    if rng.random() < 0.7:
        out = add_shadows(out, seed=seed)
    if rng.random() < 0.7:
        out = add_canopy(out, seed=seed)
    return out
