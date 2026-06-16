"""D-LinkNet-equivalent road extractor — the BASELINE segmentation backbone.

Role in the project
-------------------
This is the *baseline* binary road-segmentation network. It maps a satellite
RGB chip to a single-channel road-probability mask. The classic D-LinkNet
(DeepGlobe 2018 winner) is a LinkNet with a ResNet34 encoder plus a dilated
"center" block; here we use the LinkNet/ResNet34 implementation from
``segmentation_models_pytorch`` (smp), which is the practical, well-tested
D-LinkNet equivalent and ships ImageNet-pretrained encoder weights.

In the demo story this baseline model is the one that FRAGMENTS the road graph
under occlusion (clouds/shadows/canopy). The *robust* counterpart is the same
architecture trained with the synthetic-occlusion augmentation + consistency
loss defined in ``backend/app/services/occlusion.py`` — that augmentation is
applied to the input chips at train time while the clean mask stays the target,
teaching the net to "see through" occlusion and recover the critical links the
baseline loses.

Heavy deps (torch, segmentation_models_pytorch) are imported lazily inside
functions so this module stays import-safe on machines without them.

HARD CONSTRAINTS for this hackathon machine (Python 3.14, 4GB RTX 3050):
  * Do NOT train here — fine-tuning ResNet34/LinkNet at 1024px will not fit 4GB.
    Use small chips (e.g. 512px), mixed precision, and a tiny batch if you must.
  * Pretrained encoder weights download on first use (~85MB); the dataset itself
    (DeepGlobe/SpaceNet) is intentionally NOT downloaded here.

Pinned deps (add to ml/requirements.txt — versions known to work on py3.14):
    torch==2.6.0
    torchvision==0.21.0
    segmentation-models-pytorch==0.4.0
    numpy>=1.26

TODO (run instructions)
-----------------------
  1. python -m pip install -r ml/requirements.txt   # installs torch + smp
  2. from ml.models.dlinknet import build_dlinknet
     model = build_dlinknet(encoder_weights="imagenet").eval()
  3. Feed a normalized [B, 3, H, W] float tensor; output is [B, 1, H, W] logits.
     Apply torch.sigmoid(...) and threshold (e.g. > 0.5) to get the road mask.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # type-checker only; never imported at runtime
    import torch
    import torch.nn as nn


# --- Default architecture/config knobs --------------------------------------
DEFAULT_ENCODER: str = "resnet34"          # D-LinkNet uses a ResNet34 encoder
DEFAULT_ENCODER_WEIGHTS: str = "imagenet"  # pretrained backbone weights
IN_CHANNELS: int = 3                        # satellite RGB
CLASSES: int = 1                            # binary road / not-road (logits)


def build_dlinknet(
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: Optional[str] = DEFAULT_ENCODER_WEIGHTS,
    in_channels: int = IN_CHANNELS,
    classes: int = CLASSES,
    activation: Optional[str] = None,
    **smp_kwargs: Any,
) -> "nn.Module":
    """Build the D-LinkNet-equivalent baseline road segmentation model.

    Returns an ``smp.Linknet`` (LinkNet decoder + ResNet34 encoder) producing
    raw logits of shape ``[B, classes, H, W]``. Keep ``activation=None`` and
    apply ``torch.sigmoid`` downstream so you can use a numerically-stable
    ``BCEWithLogitsLoss`` (or Dice-on-logits) during training.

    Args:
        encoder_name: timm/smp encoder key; "resnet34" matches D-LinkNet.
        encoder_weights: "imagenet" for pretrained, or None for random init.
            (Pass None in offline/no-download environments.)
        in_channels: input channels (3 for RGB).
        classes: output channels (1 for binary roads).
        activation: smp output activation; leave None to return logits.
        **smp_kwargs: forwarded to ``smp.Linknet`` (e.g. decoder options).

    Raises:
        ImportError: if ``segmentation_models_pytorch`` (and torch) are absent;
            install them via ml/requirements.txt.
    """
    try:
        import segmentation_models_pytorch as smp  # heavy: torch under the hood
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "segmentation_models_pytorch (and torch) are required to build the "
            "D-LinkNet model. Install with: "
            "pip install -r ml/requirements.txt"
        ) from exc

    return smp.Linknet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        **smp_kwargs,
    )


def get_preprocessing_params(
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: Optional[str] = DEFAULT_ENCODER_WEIGHTS,
) -> dict[str, Any]:
    """Return the encoder's expected input-normalization params.

    Use these mean/std (matching the pretrained weights) to normalize chips
    before inference/training so the encoder sees the distribution it was
    trained on. Returns smp's preprocessing param dict (mean, std, ...).
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "segmentation_models_pytorch is required. "
            "Install with: pip install -r ml/requirements.txt"
        ) from exc

    return smp.encoders.get_preprocessing_params(encoder_name, encoder_weights)


def predict_mask(
    model: "nn.Module",
    chip: "torch.Tensor",
    threshold: float = 0.5,
) -> "torch.Tensor":
    """Run inference and return a binary road mask.

    Args:
        model: a model from :func:`build_dlinknet`.
        chip: normalized input tensor of shape ``[B, 3, H, W]`` (or ``[3, H, W]``,
            which is auto-batched). Must be on the same device as ``model``.
        threshold: sigmoid probability cutoff for the road class.

    Returns:
        A uint8 tensor of shape ``[B, H, W]`` with values in {0, 1}.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "torch is required for inference. "
            "Install with: pip install -r ml/requirements.txt"
        ) from exc

    if chip.dim() == 3:
        chip = chip.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(chip)              # [B, 1, H, W]
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).to(torch.uint8)
    return mask.squeeze(1)                  # [B, H, W]


if __name__ == "__main__":
    # Smoke test: builds the model with random init (no weight download) and
    # runs one forward pass on a synthetic chip. Requires torch + smp installed;
    # otherwise it prints a clear, actionable message and exits cleanly.
    #
    # NOTE: this is a forward-pass sanity check only — NOT training. Training the
    # baseline (and its occlusion-robust sibling) lives under ml/train/ per the
    # project plan and must be run on adequate hardware.
    try:
        import torch

        print("Building D-LinkNet-equivalent (LinkNet + ResNet34)...")
        # encoder_weights=None -> no internet/download needed for the smoke test.
        net = build_dlinknet(encoder_weights=None)
        dummy = torch.rand(1, IN_CHANNELS, 512, 512)
        out = net(dummy)
        print(f"OK: input {tuple(dummy.shape)} -> logits {tuple(out.shape)}")
        binary = predict_mask(net, dummy)
        print(f"Binary mask shape: {tuple(binary.shape)}, dtype={binary.dtype}")
    except ImportError as exc:
        print(f"[skip smoke test] {exc}")
