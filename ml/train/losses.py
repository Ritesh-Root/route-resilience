"""Segmentation + robustness losses for occlusion-robust road extraction.

This module defines the loss functions used to train the road-segmentation
network (a binary "is this pixel road?" mask per tile):

  * ``bce_dice_loss``  — BCE + (soft) Dice. BCE gives a stable per-pixel signal,
    Dice handles the thin, sparse, heavily class-imbalanced nature of roads
    (most pixels are background). Their sum is the standard, robust default for
    road/vessel/crack segmentation.

  * ``consistency_loss`` — the robustness INNOVATION. For each tile we forward
    BOTH a clean copy and a synthetically occluded copy (clouds / shadows /
    canopy from ``backend/app/services/occlusion.py``) and penalize the
    divergence between the two predictions. This trains the network to "see
    through" occlusion and produce stable masks, which is what lets the *robust*
    model recover the critical links the *baseline* loses under occlusion.

  * ``RobustSegLoss`` — convenience module bundling supervised (BCE+Dice on
    both views) + weighted consistency into the single scalar the training loop
    backprops.

Design notes
------------
* Pure PyTorch. ``torch`` is imported lazily (inside functions / a small guard)
  so this file is import-safe on machines without torch installed — handy for
  linting, doc generation and CI on the API image, which has no heavy ML deps.
* All loss fns operate on raw LOGITS (pre-sigmoid). This is numerically safer
  (``binary_cross_entropy_with_logits``) and means the model head needs no final
  activation. Pass ``logits=False`` if you already applied sigmoid.
* Targets are float masks in {0, 1} with shape broadcastable to the logits
  (``[N, 1, H, W]`` or ``[N, H, W]``).

Usage (TODO: wire into ``ml/train/train.py``)
---------------------------------------------
    # requirements (pin in ml/requirements.txt):
    #   torch==2.5.1            # CUDA 12.1 build for the RTX 3050 4GB
    #   numpy==2.1.*
    #
    # NOTE: do NOT pip install torch in this hackathon environment — these are
    # the versions the training box should use. Training runs offline.
    #
    # >>> from ml.train.losses import RobustSegLoss
    # >>> criterion = RobustSegLoss(consistency_weight=1.0)
    # >>> for clean, occ, target in loader:        # occ = random_occlude(clean)
    # ...     logits_clean = model(clean)
    # ...     logits_occ   = model(occ)
    # ...     loss, parts  = criterion(logits_clean, logits_occ, target)
    # ...     loss.backward(); opt.step(); opt.zero_grad()

Run ``python -m ml.train.losses`` for a tiny self-test on random tensors
(skips gracefully with a clear message if torch is absent).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints; never imported at runtime
    import torch
    from torch import Tensor


def _require_torch():
    """Import torch lazily so the module is import-safe without it.

    Returns the ``torch`` module. Raises a clear, actionable error otherwise
    instead of a bare ``ModuleNotFoundError`` deep in a stack trace.
    """
    try:
        import torch  # noqa: F401  (returned to caller)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "losses.py needs PyTorch at call time. Install the CUDA 12.1 build "
            "matching the RTX 3050 (e.g. `pip install torch==2.5.1 "
            "--index-url https://download.pytorch.org/whl/cu121`). "
            "Do NOT install it in the API/demo environment — training is "
            "expected to run on the dedicated GPU box only."
        ) from exc
    return torch


# --------------------------------------------------------------------------- #
# Core supervised losses
# --------------------------------------------------------------------------- #
def dice_loss(
    logits: "Tensor",
    target: "Tensor",
    *,
    smooth: float = 1.0,
    from_logits: bool = True,
) -> "Tensor":
    """Soft Dice loss for binary segmentation: ``1 - Dice``.

    Dice = ``2|P∩T| / (|P| + |T|)``; the soft version uses probabilities so it
    is differentiable. The ``smooth`` term avoids 0/0 on empty tiles and acts as
    mild Laplace smoothing. Robust to the strong fg/bg imbalance of roads.

    Args:
        logits: Raw model output ``[N, 1, H, W]`` (or ``[N, H, W]``).
        target: Ground-truth mask in {0, 1}, broadcastable to ``logits``.
        smooth: Numerical-stability / smoothing constant.
        from_logits: If True, apply sigmoid to ``logits`` first.

    Returns:
        Scalar tensor (mean Dice loss over the batch).
    """
    torch = _require_torch()
    probs = torch.sigmoid(logits) if from_logits else logits
    target = target.to(probs.dtype)

    # Flatten everything except the batch dim so Dice is computed per-sample.
    n = probs.shape[0]
    p = probs.reshape(n, -1)
    t = target.reshape(n, -1)

    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def bce_loss(
    logits: "Tensor",
    target: "Tensor",
    *,
    from_logits: bool = True,
    pos_weight: "Tensor | float | None" = None,
) -> "Tensor":
    """Binary cross-entropy. Uses the logits-stable form when ``from_logits``.

    Args:
        logits: Raw model output (or probabilities if ``from_logits`` is False).
        target: Ground-truth mask in {0, 1}, same broadcast shape as ``logits``.
        from_logits: If True, use ``binary_cross_entropy_with_logits``.
        pos_weight: Optional positive-class weight to up-weight sparse road
            pixels (recall). Scalar or tensor broadcastable over the class dim.

    Returns:
        Scalar tensor (mean BCE).
    """
    torch = _require_torch()
    import torch.nn.functional as F

    target = target.to(logits.dtype)
    if from_logits:
        pw = None
        if pos_weight is not None:
            pw = (
                pos_weight
                if isinstance(pos_weight, torch.Tensor)
                else torch.as_tensor(
                    pos_weight, dtype=logits.dtype, device=logits.device
                )
            )
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pw
        )
    # logits already in [0, 1]; clamp to avoid log(0).
    probs = logits.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(probs, target)


def bce_dice_loss(
    logits: "Tensor",
    target: "Tensor",
    *,
    bce_w: float = 1.0,
    dice_w: float = 1.0,
    smooth: float = 1.0,
    from_logits: bool = True,
    pos_weight: "Tensor | float | None" = None,
) -> "Tensor":
    """Weighted sum of BCE and soft Dice — the default road-seg objective.

    BCE supplies a dense, well-behaved gradient everywhere; Dice directly
    optimizes overlap and counteracts class imbalance on thin roads. TODO: add
    a focal term here if recall on thin/occluded roads still lags (see
    architecture notes).

    Returns:
        Scalar tensor: ``bce_w * BCE + dice_w * Dice``.
    """
    bce = bce_loss(
        logits, target, from_logits=from_logits, pos_weight=pos_weight
    )
    dice = dice_loss(logits, target, smooth=smooth, from_logits=from_logits)
    return bce_w * bce + dice_w * dice


# --------------------------------------------------------------------------- #
# Robustness: clean vs. occluded consistency  (the headline innovation)
# --------------------------------------------------------------------------- #
def consistency_loss(
    logits_clean: "Tensor",
    logits_occluded: "Tensor",
    *,
    mode: str = "mse",
    from_logits: bool = True,
    occlusion_mask: "Tensor | None" = None,
) -> "Tensor":
    """Penalize divergence between predictions on clean vs. occluded tiles.

    Both inputs are predictions for the SAME geographic tile — one from the
    clean chip and one from its occluded copy produced by
    ``backend/app/services/occlusion.py`` (clouds / shadows / canopy). Driving
    the two predictions together teaches the network to ignore the occluder and
    keep the underlying road graph intact.

    We compare in PROBABILITY space (sigmoid) so the penalty is bounded and
    well-scaled regardless of logit magnitude.

    Args:
        logits_clean: Prediction on the clean view.
        logits_occluded: Prediction on the occluded view (same tile/order).
        mode: ``"mse"`` (symmetric L2 on probabilities, simplest/stable) or
            ``"kl"`` (symmetric KL treating each pixel as a Bernoulli; sharper
            signal where the two disagree confidently).
        from_logits: If True, sigmoid the inputs before comparing.
        occlusion_mask: Optional float mask (same spatial shape, broadcastable)
            in [0, 1] marking occluded pixels. When given, the loss is averaged
            only over occluded regions — focusing the consistency pressure where
            the occluder actually hides road. Mirrors the soft blob masks in
            ``occlusion.py``. If the mask is all-zero (no occlusion this batch)
            the loss is 0.
    """
    torch = _require_torch()

    if mode not in ("mse", "kl"):
        raise ValueError(f"mode must be 'mse' or 'kl', got {mode!r}")

    if from_logits:
        p_clean = torch.sigmoid(logits_clean)
        p_occ = torch.sigmoid(logits_occluded)
    else:
        p_clean = logits_clean
        p_occ = logits_occluded

    # Treat the clean prediction as the (stop-grad) target the occluded one
    # should match. Detaching prevents the easy degenerate solution where the
    # model corrupts the clean prediction to meet the occluded one halfway.
    p_clean_t = p_clean.detach()

    if mode == "mse":
        per_pixel = (p_occ - p_clean_t) ** 2
    else:  # symmetric Bernoulli KL between the two pixel-wise distributions
        eps = 1e-6
        pc = p_clean_t.clamp(eps, 1.0 - eps)
        po = p_occ.clamp(eps, 1.0 - eps)
        kl = (
            po * (po / pc).log()
            + (1 - po) * ((1 - po) / (1 - pc)).log()
            + pc * (pc / po).log()
            + (1 - pc) * ((1 - pc) / (1 - po)).log()
        )
        per_pixel = 0.5 * kl

    if occlusion_mask is not None:
        m = occlusion_mask.to(per_pixel.dtype)
        denom = m.sum().clamp_min(1.0)
        return (per_pixel * m).sum() / denom

    return per_pixel.mean()


# --------------------------------------------------------------------------- #
# Bundled training objective
# --------------------------------------------------------------------------- #
def _build_module_base():
    """Return ``torch.nn.Module`` to subclass, importing torch lazily.

    Kept in a helper so the class body below can reference it without a
    module-level torch import (which would break import-safety).
    """
    torch = _require_torch()
    return torch.nn.Module


class RobustSegLoss:
    """Supervised BCE+Dice on both views + weighted clean/occluded consistency.

    Not a ``nn.Module`` subclass (so the file imports without torch); it holds
    only plain Python hyper-parameters and calls the functional losses above.
    This is sufficient because it has no learnable parameters or buffers.

    Total loss::

        L = sup_w  * [ BCEDice(clean, target) + BCEDice(occluded, target) ] / 2
          + cons_w * consistency(clean, occluded)

    Including the supervised term on the OCCLUDED view (not just consistency)
    keeps the occluded prediction anchored to ground truth, while consistency
    ties it to the (easier) clean prediction.

    Args:
        bce_w / dice_w: Weights inside each BCE+Dice term.
        supervised_weight: Scale on the averaged supervised term.
        consistency_weight: Scale on the consistency term (the robustness knob;
            TODO ramp from 0 over the first few epochs for stability).
        consistency_mode: ``"mse"`` or ``"kl"`` (see ``consistency_loss``).
        pos_weight: Optional positive-class weight forwarded to BCE.
        smooth: Dice smoothing constant.
    """

    def __init__(
        self,
        *,
        bce_w: float = 1.0,
        dice_w: float = 1.0,
        supervised_weight: float = 1.0,
        consistency_weight: float = 1.0,
        consistency_mode: str = "mse",
        pos_weight: "Tensor | float | None" = None,
        smooth: float = 1.0,
    ) -> None:
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.supervised_weight = supervised_weight
        self.consistency_weight = consistency_weight
        self.consistency_mode = consistency_mode
        self.pos_weight = pos_weight
        self.smooth = smooth

    def __call__(
        self,
        logits_clean: "Tensor",
        logits_occluded: "Tensor",
        target: "Tensor",
        *,
        from_logits: bool = True,
        occlusion_mask: "Tensor | None" = None,
    ) -> "tuple[Tensor, dict[str, Tensor]]":
        """Compute the total loss and a breakdown dict (for logging).

        Returns:
            ``(total, parts)`` where ``parts`` has keys
            ``sup_clean``, ``sup_occluded``, ``consistency``, ``total`` —
            handy for TensorBoard / W&B curves.
        """
        sup_clean = bce_dice_loss(
            logits_clean,
            target,
            bce_w=self.bce_w,
            dice_w=self.dice_w,
            smooth=self.smooth,
            from_logits=from_logits,
            pos_weight=self.pos_weight,
        )
        sup_occ = bce_dice_loss(
            logits_occluded,
            target,
            bce_w=self.bce_w,
            dice_w=self.dice_w,
            smooth=self.smooth,
            from_logits=from_logits,
            pos_weight=self.pos_weight,
        )
        cons = consistency_loss(
            logits_clean,
            logits_occluded,
            mode=self.consistency_mode,
            from_logits=from_logits,
            occlusion_mask=occlusion_mask,
        )

        sup = self.supervised_weight * 0.5 * (sup_clean + sup_occ)
        total = sup + self.consistency_weight * cons
        return total, {
            "sup_clean": sup_clean,
            "sup_occluded": sup_occ,
            "consistency": cons,
            "total": total,
        }


# --------------------------------------------------------------------------- #
# Tiny self-test  (no data, no training — just verifies the math runs)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch = _require_torch()
    torch.manual_seed(0)

    n, h, w = 2, 1, 16  # [N, C, H, W] -> use C=1
    logits_clean = torch.randn(n, 1, w, w, requires_grad=True)
    # Occluded prediction = clean + perturbation, as a stand-in for a 2nd pass.
    logits_occ = logits_clean.detach() + 0.5 * torch.randn(n, 1, w, w)
    logits_occ.requires_grad_(True)
    target = (torch.rand(n, 1, w, w) > 0.7).float()  # sparse "roads"

    print("dice_loss      :", float(dice_loss(logits_clean, target)))
    print("bce_loss       :", float(bce_loss(logits_clean, target)))
    print("bce_dice_loss  :", float(bce_dice_loss(logits_clean, target)))
    print("consistency mse:", float(consistency_loss(logits_clean, logits_occ)))
    print("consistency kl :",
          float(consistency_loss(logits_clean, logits_occ, mode="kl")))

    # Masked consistency (only occluded pixels count).
    occ_mask = (torch.rand(n, 1, w, w) > 0.5).float()
    print("consistency msk:",
          float(consistency_loss(logits_clean, logits_occ,
                                 occlusion_mask=occ_mask)))

    crit = RobustSegLoss(consistency_weight=1.0)
    total, parts = crit(logits_clean, logits_occ, target)
    total.backward()  # confirms the full graph is differentiable end-to-end
    print("RobustSegLoss  :", {k: round(float(v), 4) for k, v in parts.items()})
    assert logits_clean.grad is not None and logits_occ.grad is not None
    print("OK: gradients flow through both views.")


if __name__ == "__main__":  # pragma: no cover
    try:
        _selftest()
    except ImportError as exc:
        # Import-safe path: torch absent (e.g. on the API/demo box). This is
        # expected here — training runs on the dedicated GPU machine.
        print(f"[losses self-test skipped] {exc}")
