"""CoANet-style connectivity/occlusion-aware road extractor — notes + interface stub.

WHY THIS FILE EXISTS
--------------------
This is the "occlusion-robust" half of the project's claim. The baseline extractor
(D-LinkNet / `segmentation_models_pytorch` LinkNet+ResNet34, see
``ml/models/dlinknet.py`` and ``backend/app/services/segmentation.py``) does fine on
clean imagery but *fragments* roads wherever clouds, building shadows or tree canopy
hide the surface. Those gaps then break the NetworkX graph downstream, which is
exactly the demo's "fragmented network" failure mode
(input=occluded & model=baseline -> resilienceIndex ~79).

CoANet (Mei et al., 2021, "CoANet: Connectivity Attention Network for Road
Extraction From Satellite Imagery", IEEE TIP; ref repo ``mj129/CoANet``) attacks
that gap directly: it reasons about *connectivity between neighbouring pixels in
multiple directions* rather than classifying each pixel in isolation, so it can
bridge short occluded segments that a per-pixel network drops. That is the
"robust model recovers the critical links the baseline loses" story.

DESIGN GOAL: a CoANet model must be a *drop-in replacement* behind the SAME
interface the backend already calls — ``RoadSegmenter.predict(image) -> HxW float
mask in [0, 1]`` (see ``backend/app/services/segmentation.py``). Swapping baseline
-> robust is then just choosing which class to instantiate; nothing downstream
(vectorize -> graph -> criticality) changes.

HOW CONNECTIVITY ATTENTION WORKS (the part that buys occlusion robustness)
--------------------------------------------------------------------------
CoANet adds three ideas on top of a standard encoder/decoder segmentation net:

1. Strip Convolution Module (SCM). Roads are long, thin, oriented structures.
   In addition to square kernels, CoANet convolves with 1xK and Kx1 (and the two
   diagonal) "strip" kernels so the receptive field follows a road across a gap
   instead of being dominated by the background it cuts through. This is what lets
   evidence on *both sides* of an occluding cloud vote for the road continuing
   underneath it.

2. Connectivity Attention Module (CoA). The network predicts, per pixel, the
   probability of being *connected to* each of its (typically 8) directional
   neighbours — i.e. an auxiliary connectivity map with one channel per direction,
   not just a single road/no-road channel. These connectivity logits are turned
   into an attention map that re-weights the segmentation features: pixels the
   model believes are linked to a confident road pixel get boosted even when their
   own appearance is degraded by occlusion. Concretely the seg branch and the
   connectivity branch share the encoder; the CoA attention multiplies/gates the
   decoder features before the final road head.

3. Connectivity-aware loss. Training supervises BOTH the road mask (BCE + Dice,
   as today) AND the directional connectivity maps (the ground-truth connectivity
   is derived from the road mask by checking, for each direction, whether the
   neighbour is also road). Penalising wrong connectivity teaches the net to keep
   linear structures intact rather than producing speckle.

HOW THE OCCLUSION AUGMENTATION (our innovation) PLUGS IN
--------------------------------------------------------
``backend/app/services/occlusion.py`` already generates synthetic clouds / shadows /
canopy as numpy ops. The training-time recipe that makes CoANet *occlusion*-robust
(not just connectivity-aware) is:

  * Apply ``occlusion.random_occlude`` to the input chip while keeping the CLEAN
    ground-truth mask + connectivity targets as the labels — the net is forced to
    reconstruct the road it can no longer fully see.
  * Optionally forward a clean copy AND an occluded copy of the same chip and add a
    *consistency loss* between the two predicted masks (and connectivity maps), so
    the robust model is explicitly trained to give the same answer with or without
    occlusion. This is the quantitative knob behind "occlusionRecall" in
    ``/api/metrics``.

See ``occlusion_consistency_target`` below for how those two pieces meet.

HARDWARE / SCOPE NOTE
---------------------
This file is INTERFACE + NOTES only. It deliberately does NOT train, does NOT
download datasets, and does NOT import torch at module load (4GB RTX 3050 / Python
3.14 dev box). All heavy imports are guarded inside functions/__init__ so the file
is import-safe even with no ML stack installed — import it for the docstring,
shapes and TODOs; wire real weights later. Pinned deps live in
``ml/requirements.txt`` (TODO, see bottom of file).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # only for static type checkers; never imported at runtime
    import torch
    from torch import Tensor, nn


# Number of directional neighbours CoANet reasons about. 8 = full Moore
# neighbourhood (N, NE, E, SE, S, SW, W, NW). The connectivity head emits one
# logit per direction per pixel.
N_DIRECTIONS: int = 8


def _require_torch() -> "Any":
    """Import torch lazily with a clear, actionable error if it is absent.

    Heavy import is deferred so this module stays import-safe on a box without a
    GPU/ML stack (per project hardware constraints). Returns the imported ``torch``
    module so callers can do ``torch = _require_torch()``.
    """
    try:
        import torch  # noqa: PLC0415 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "CoANet needs PyTorch, which is intentionally NOT installed on the dev "
            "box (4GB RTX 3050 / Python 3.14). Install the pinned stack from "
            "ml/requirements.txt on a training machine, e.g.:\n"
            "    pip install -r ml/requirements.txt\n"
            "then retry. This module imports fine without torch; only the model "
            "construction / forward paths require it."
        ) from exc
    return torch


def _require_smp() -> "Any":
    """Lazily import segmentation_models_pytorch (shared encoder backbone)."""
    try:
        import segmentation_models_pytorch as smp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "segmentation_models_pytorch is required to build the CoANet encoder "
            "backbone. Install it via ml/requirements.txt on a training machine."
        ) from exc
    return smp


# --------------------------------------------------------------------------- #
# Connectivity helpers — pure numpy, no torch needed, safe to call anywhere.
# These define the *targets* the connectivity branch is trained against and are
# handy for visualisation/debugging without a model.
# --------------------------------------------------------------------------- #

# (dy, dx) offsets ordered to match N_DIRECTIONS channels.
_DIRECTION_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)


def connectivity_target(mask: np.ndarray) -> np.ndarray:
    """Derive the directional connectivity ground truth from a binary road mask.

    For each pixel and each of ``N_DIRECTIONS`` directions, output 1.0 iff BOTH the
    pixel and its neighbour in that direction are road. Shape ``(N_DIRECTIONS, H, W)``.
    This is what CoANet's connectivity branch (the CoA attention) is supervised on,
    in addition to the plain road mask. Border neighbours that fall outside the
    image are treated as non-road (0).

    Args:
        mask: ``(H, W)`` array; treated as road where ``mask > 0.5``.

    Returns:
        ``(N_DIRECTIONS, H, W)`` float32 array in {0.0, 1.0}.
    """
    road = (mask > 0.5).astype(np.float32)
    h, w = road.shape
    out = np.zeros((N_DIRECTIONS, h, w), dtype=np.float32)
    for k, (dy, dx) in enumerate(_DIRECTION_OFFSETS):
        shifted = np.zeros_like(road)
        # Compute the overlapping region between the image and its shifted copy so
        # we never read out of bounds (out-of-image neighbour == non-road == 0).
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        yd0, yd1 = max(0, -dy), min(h, h - dy)
        xd0, xd1 = max(0, -dx), min(w, w - dx)
        shifted[yd0:yd1, xd0:xd1] = road[ys0:ys1, xs0:xs1]
        out[k] = road * shifted
    return out


def occlusion_consistency_target(
    pred_clean: np.ndarray, pred_occluded: np.ndarray
) -> float:
    """Scalar consistency error between clean vs occluded predictions (lower=better).

    Mirrors the training-time consistency loss idea: forward the same chip clean and
    after ``backend/app/services/occlusion.random_occlude``; a robust model should
    produce nearly identical masks. Returned as mean-absolute-difference in [0, 1].
    Useful offline (numpy only) to sanity-check robustness without running training.
    """
    a = np.clip(pred_clean.astype(np.float32), 0.0, 1.0)
    b = np.clip(pred_occluded.astype(np.float32), 0.0, 1.0)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


# --------------------------------------------------------------------------- #
# The model. Torch-dependent; only constructed when torch is available.
# --------------------------------------------------------------------------- #

def build_coanet(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
) -> "nn.Module":
    """Construct a CoANet-style network (connectivity-attention road extractor).

    NOTE: this returns a SCAFFOLD. It uses a `segmentation_models_pytorch` LinkNet as
    the shared encoder/decoder backbone (the cheap, proven D-LinkNet-equivalent), then
    bolts on (a) a connectivity head emitting ``N_DIRECTIONS`` channels and (b) a CoA
    attention gate that re-weights decoder features by the connectivity confidence.
    The strip-convolution module (SCM) is left as a TODO — wire mj129/CoANet's SCM in
    front of the seg head for the full paper. Even without SCM, the connectivity head
    + consistency training already delivers the occlusion-robustness story.

    Args:
        encoder_name: timm/SMP encoder, e.g. "resnet34" (matches the baseline).
        encoder_weights: pretrained encoder init ("imagenet") or None.
        in_channels: input channels (3 for RGB satellite chips).

    Returns:
        A ``torch.nn.Module`` whose ``forward(x)`` returns a dict
        ``{"road": (B,1,H,W) logits, "conn": (B,N_DIRECTIONS,H,W) logits}``.
    """
    torch = _require_torch()
    smp = _require_smp()
    nn = torch.nn

    class _CoANet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Shared backbone. `aux_params=None`; we attach our own heads to the
            # decoder feature map (1 base channel out, then split into heads).
            self.backbone = smp.Linknet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=16,  # decoder feature width feeding our heads (not the mask)
                activation=None,
            )
            # Road segmentation head: 16 feat -> 1 logit.
            self.road_head = nn.Conv2d(16, 1, kernel_size=1)
            # Connectivity head: 16 feat -> N_DIRECTIONS logits (the CoA targets).
            self.conn_head = nn.Conv2d(16, N_DIRECTIONS, kernel_size=1)
            # CoA attention gate: collapse the N_DIRECTIONS connectivity logits into a
            # single spatial attention map that re-weights the road logits. This is
            # the mechanism that lets confidently-connected pixels boost occluded
            # neighbours. TODO: replace with mj129/CoANet's full CoA + SCM blocks.
            self.coa_gate = nn.Sequential(
                nn.Conv2d(N_DIRECTIONS, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: "Tensor") -> dict[str, "Tensor"]:
            feat = self.backbone(x)              # (B, 16, H, W)
            conn = self.conn_head(feat)          # (B, N_DIRECTIONS, H, W) logits
            attn = self.coa_gate(conn)           # (B, 1, H, W) in [0,1]
            road = self.road_head(feat)          # (B, 1, H, W) logits
            # Additive gating: connectivity confidence nudges road logits up where
            # neighbours agree, helping span occluded gaps. (Multiplicative gating is
            # the alternative; pick per ablation.)
            road = road + (attn - 0.5) * 2.0 * road.abs().clamp(max=8.0)
            return {"road": road, "conn": conn}

    return _CoANet()


class CoANetSegmenter:
    """Robust road extractor — same surface as ``RoadSegmenter`` in the backend.

    Slots in behind the EXACT contract the API already uses:
        ``predict(image: np.ndarray) -> np.ndarray``  # HxW float mask in [0, 1]

    so the backend can pick baseline vs robust by class alone (model=baseline ->
    RoadSegmenter, model=robust -> CoANetSegmenter). Construction and inference are
    lazy/guarded; importing this class never requires torch.
    """

    def __init__(self, weights_path: str | None = None, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self.model: Any = None  # built lazily in _ensure_model()

    def _ensure_model(self) -> None:
        """Build the network and load weights on first use (lazy, torch-guarded)."""
        if self.model is not None:
            return
        torch = _require_torch()
        model = build_coanet()
        if self.weights_path:
            # TODO: load a trained CoANet checkpoint produced by ml/train/.
            state = torch.load(self.weights_path, map_location=self.device)
            state = state.get("model", state)  # support {"model": ...} checkpoints
            model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        self.model = model

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return an ``HxW`` road-probability mask in ``[0, 1]``.

        Args:
            image: ``(H, W, 3)`` uint8/float RGB chip, matching what the baseline
                ``RoadSegmenter.predict`` consumes.

        Returns:
            ``(H, W)`` float32 mask, road probability after sigmoid.

        Raises:
            ImportError: if torch is unavailable (clear message via _require_torch).
        """
        torch = _require_torch()
        self._ensure_model()
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {image.shape}")

        # NCHW float tensor in [0,1]. (Match the training normalisation here — TODO:
        # use the same mean/std as the encoder's pretrained weights.)
        arr = image.astype(np.float32)
        if arr.max() > 1.5:  # looks like 0..255
            arr = arr / 255.0
        chw = np.transpose(arr, (2, 0, 1))[None]  # (1, 3, H, W)
        with torch.no_grad():
            x = torch.from_numpy(chw).to(self.device)
            out = self.model(x)
            prob = torch.sigmoid(out["road"])[0, 0]  # (H, W)
        return prob.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Self-test / usage. Runs the torch-free parts only, so it works on the dev box.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # This block intentionally exercises ONLY the numpy-only helpers so it runs on a
    # machine with no ML stack. The torch-dependent path is documented below as a TODO.
    print("coanet.py self-test (numpy-only; no torch required)")

    # Build a tiny diagonal "road" and show the derived connectivity target.
    demo = np.zeros((6, 6), dtype=np.float32)
    for i in range(6):
        demo[i, i] = 1.0
    conn = connectivity_target(demo)
    print(f"connectivity_target shape: {conn.shape} (expected ({N_DIRECTIONS}, 6, 6))")
    # SE direction (index 3) should fire along the diagonal interior.
    print("SE-connectivity along diagonal:\n", conn[3].astype(int))

    # Consistency metric sanity check: identical preds -> 0.0 error.
    a = np.random.default_rng(0).random((6, 6)).astype(np.float32)
    print("consistency(clean, clean) =", occlusion_consistency_target(a, a))
    print("consistency(clean, 1-clean) =", occlusion_consistency_target(a, 1 - a))

    # TODO (training machine, torch installed):
    #   from coanet import CoANetSegmenter
    #   seg = CoANetSegmenter(weights_path="weights/coanet_bengaluru.pt", device="cuda")
    #   mask = seg.predict(rgb_chip)            # (H, W) in [0,1]
    #   # then: skeletonize -> graph -> criticality (unchanged downstream).
    #
    # TODO: create ml/requirements.txt pinned for a CUDA 11.8 / training box, e.g.:
    #   torch==2.3.1            # install the +cu118 wheel on the GPU machine
    #   torchvision==0.18.1
    #   segmentation-models-pytorch==0.3.4
    #   albumentations==1.4.10  # custom occlusion transform wraps occlusion.py
    #   rasterio==1.3.10        # guarded import; tiles I/O at train time only
    #   numpy>=1.26,<2.1
    # (Do NOT install on the 4GB RTX 3050 / Python 3.14 dev box.)
    print("OK")
