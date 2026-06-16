"""Segmentation interface (stub).

The demo serves a precomputed synthetic network so the API works today. When the
trained model is ready, implement `RoadSegmenter.predict` to return a road
probability mask, then run it through vectorization (skeletonize -> graph) and
hand the resulting NetworkX graph to the criticality engine.

Recommended implementation (see PROJECT-PLAN.md):
    encoder = ResNet34 (ImageNet)            # via segmentation_models_pytorch
    arch    = LinkNet/DLinkNet  OR  CoANet   # connectivity-aware option
    loss    = BCE + Dice (+ occlusion consistency)
"""
from __future__ import annotations

import numpy as np


class RoadSegmenter:
    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path
        self.model = None  # TODO: load torch model + weights

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return HxW float mask in [0,1]. Not implemented in the demo."""
        raise NotImplementedError(
            "Wire a trained D-LinkNet/CoANet checkpoint here. "
            "Until then the API serves the synthetic network from network_factory."
        )
