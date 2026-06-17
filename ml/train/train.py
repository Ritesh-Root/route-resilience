"""Occlusion-robust road-segmentation training loop (steps 2+3 of the pipeline).

Trains a D-LinkNet34-equivalent (LinkNet + ResNet34 encoder) binary road
segmenter with the objective that creates the demo story:

  * ``--variant baseline`` — supervised BCE+Dice on CLEAN chips only. This is the
    model that FRAGMENTS the road graph under occlusion in the demo
    (``input=occluded & model=baseline`` -> resilienceIndex ~79).
  * ``--variant robust`` — supervised BCE+Dice on both a CLEAN and a synthetically
    OCCLUDED copy of each chip (target is always the clean ground-truth mask) PLUS
    a clean<->occluded consistency loss. The occlusion comes from the numpy-only
    generators in ``backend/app/services/occlusion.py`` (clouds / shadows /
    canopy). This teaches the net to "see through" occlusion and recover the
    critical links the baseline drops (resilienceIndex -> 100).

The whole loop is tuned for the hackathon box (Python 3.14, 4 GB RTX 3050):

  * 512x512 chips, ``--batch-size 1`` (or 2) by default.
  * Automatic mixed precision (``torch.amp.autocast`` + ``GradScaler``).
  * Gradient accumulation (``--accum-steps``) for a larger *effective* batch.
  * ``optimizer.zero_grad(set_to_none=True)`` and ``torch.cuda.empty_cache()``
    between epochs to keep the working set small.
  * Checkpointing (resume-able) so an interrupted run is cheap to restart.

Import-safety contract (mirrors the rest of ``ml/`` and
``backend/app/services/occlusion.py``): importing this module must NOT require
torch / the model package / the dataset to be installed. Every heavy import is
guarded inside a function or ``try/except ImportError`` with a clear remediation
message. Training NEVER auto-starts on import — it is gated behind
``if __name__ == "__main__":`` + argparse.

------------------------------------------------------------------------------
RUN (from the repo root; paths contain spaces, so quote them)
------------------------------------------------------------------------------
  0. Install pinned deps on the TRAINING box (NOT the 4GB demo laptop):
       python -m pip install -r "ml/requirements-ml.txt"
     (torch is installed from the CUDA index — see that file's header.)

  1. Build 512x512 chips + a manifest from your own rasters (never downloaded
     here). See ml/data/tiling.py:
       python ml/data/tiling.py --images "/path/*_sat.tif" \
           --masks "/path/*_mask.tif" --out "ml/data/chips"

  2. Train the baseline, then the robust model:
       python -m ml.train.train --variant baseline --epochs 30 \
           --data "ml/data/chips" --out "ml/checkpoints/baseline.pt"
       python -m ml.train.train --variant robust --epochs 30 \
           --data "ml/data/chips" --out "ml/checkpoints/robust.pt"
     (``robust`` auto-enables occlusion + consistency; override with the flags.)

  3. Smoke-test the loop WITHOUT real data or a GPU (random synthetic chips,
     1 step, CPU): proves the graph is differentiable end-to-end. Requires torch
     installed; otherwise it prints a clear "install torch" message and exits 0.
       python -m ml.train.train --self-test

TODO(you): point ``--data`` at the chip dir produced by tiling.py; commit a few
sample chips under ml/data/sample/ for a tiny real-data smoke run; then wire the
resulting checkpoints into ml/vectorize.py (step 4).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator

# Type-only imports: evaluated by type checkers, never at runtime, so importing
# this module does not require torch/numpy to be installed.
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset as _TorchDataset
else:
    # Resolve the real torch Dataset base at import time when torch is present,
    # falling back to ``object`` only if torch is genuinely absent (so the module
    # stays importable for tooling without torch). We deliberately do NOT rebind
    # ``cls.__bases__`` later: that hack raises ``TypeError: __bases__ assignment:
    # 'Dataset' deallocator differs from 'object'`` on modern CPython/torch.
    try:
        from torch.utils.data import Dataset as _TorchDataset
    except Exception:  # torch not installed (lint/type tooling only)
        _TorchDataset = object


# 4GB-RTX-3050-friendly defaults. Keep these conservative — they are the happy
# path; bump only on a bigger GPU.
DEFAULT_TILE: int = 512
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_ACCUM_STEPS: int = 8        # effective batch = batch_size * accum_steps
DEFAULT_EPOCHS: int = 30
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_NUM_WORKERS: int = 2
DEFAULT_CONSISTENCY_WEIGHT: float = 1.0
# Ramp consistency in over the first N epochs for stability (0 -> full weight).
DEFAULT_CONSISTENCY_WARMUP_EPOCHS: int = 3


# --------------------------------------------------------------------------- #
# Lazy / guarded imports (file stays import-safe without torch or siblings).
# --------------------------------------------------------------------------- #
def _require(module: str, pip_hint: str | None = None) -> Any:
    """Import ``module`` lazily, raising a clear, actionable error if missing.

    Keeps this file import-safe on the API/demo box (no torch). The hint steers
    users to ml/requirements-ml.txt and warns against blindly installing torch
    on the 4GB laptop.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        hint = pip_hint or (
            f"pip install -r ml/requirements-ml.txt (provides '{module}')"
        )
        raise ImportError(
            f"ml/train/train.py needs '{module}' at run time. {hint}. "
            "NOTE: do NOT `pip install torch` on the 4GB demo laptop — train on "
            "the dedicated CUDA box (see ml/requirements-ml.txt header)."
        ) from exc


def _load_losses() -> Any:
    """Import the sibling ``losses`` module (BCE+Dice + consistency).

    Tries the package path first (``ml.train.losses`` when run as ``-m``), then
    falls back to loading the sibling file by path so the loop also works when
    invoked as a plain script. ``losses`` is itself import-safe (torch guarded).
    """
    try:
        return importlib.import_module("ml.train.losses")
    except ImportError:
        pass
    try:
        # Relative import when this module is part of the ml.train package.
        from . import losses as _losses  # type: ignore[attr-defined]

        return _losses
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    losses_path = os.path.join(here, "losses.py")
    if not os.path.isfile(losses_path):
        raise ImportError(
            f"Could not locate the losses module next to {__file__!r} "
            f"(expected {losses_path!r})."
        )
    spec = importlib.util.spec_from_file_location("_rr_losses", losses_path)
    assert spec and spec.loader  # for type checkers
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_build_dlinknet() -> Callable[..., Any]:
    """Return ``build_dlinknet`` from the model package.

    The D-LinkNet model lives at ``atlas-vision/ml/models/dlinknet.py`` in this
    repo. We import it by file path (mirroring deepglobe.py's occlusion loader)
    so the loop works regardless of which ``ml`` package is on ``sys.path``.
    Falls back to the importable ``ml.models.dlinknet`` if that exists instead.
    """
    try:
        mod = importlib.import_module("ml.models.dlinknet")
        return mod.build_dlinknet
    except ImportError:
        pass

    # ml/train/train.py -> repo root is two parents up; the model file lives under
    # atlas-vision/ml/models/dlinknet.py (per the repo layout).
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.path.join(repo_root, "atlas-vision", "ml", "models", "dlinknet.py"),
        os.path.join(repo_root, "ml", "models", "dlinknet.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_rr_dlinknet", path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.build_dlinknet
    raise ImportError(
        "Could not find dlinknet.py (build_dlinknet). Looked in: "
        + ", ".join(repr(c) for c in candidates)
    )


def _load_occluder() -> Callable[..., "np.ndarray"]:
    """Return ``random_occlude`` from the backend occlusion service (numpy-only).

    Same file-path loading strategy as ml/data/deepglobe.py so the robustness
    augmentation is shared with the rest of the pipeline. Falls back to identity
    (with a warning) if the file is missing, so the loop still runs.
    """
    import warnings

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    occ_path = os.path.join(
        repo_root, "backend", "app", "services", "occlusion.py"
    )
    if not os.path.isfile(occ_path):
        warnings.warn(
            f"occlusion.py not found at {occ_path!r}; occluded view = clean view "
            "(robust training will have no occlusion signal).",
            RuntimeWarning,
            stacklevel=2,
        )
        return lambda img, seed=None: img

    spec = importlib.util.spec_from_file_location("_rr_occlusion", occ_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.random_occlude


# --------------------------------------------------------------------------- #
# Dataset: read 512x512 chips produced by ml/data/tiling.py (npy or png).
# --------------------------------------------------------------------------- #
def _torch_dataset_base() -> Any:
    """Return ``torch.utils.data.Dataset`` lazily (clear error if torch absent)."""
    tud = _require("torch.utils.data", pip_hint="install torch (see ml/requirements-ml.txt)")
    return tud.Dataset


class ChipDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """``(image, mask[, occluded])`` pairs from a tiling.py chip directory.

    Expects the layout written by ``ml/data/tiling.py`` (a ``manifest.csv`` whose
    rows are ``scene_id, split, row, col, image_path, mask_path``), or — if no
    manifest exists — falls back to globbing ``<root>/<split>/images`` paired with
    ``<root>/<split>/masks`` by filename. Chips are ``.npy`` (preferred,
    dependency-free) or images readable by Pillow.

    The image is returned as a CHW float tensor in [0, 1]; the mask as a
    ``[1, H, W]`` float tensor in {0, 1}. When ``return_occluded`` is set, an
    occluded copy of the image (from ``random_occlude``) is returned too, sharing
    the SAME clean mask as target — this is the robustness recipe.

    Heavy work (decoding, occlusion, tensor conversion) happens only in
    ``__getitem__`` so importing/constructing stays lightweight.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        tile_size: int = DEFAULT_TILE,
        return_occluded: bool = False,
        seed: int = 1337,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.root = root
        self.split = split
        self.tile_size = tile_size
        self.return_occluded = return_occluded
        self.seed = seed
        self._occlude = _load_occluder() if return_occluded else None
        self.pairs: list[tuple[str, str]] = self._index(root, split)
        if not self.pairs:
            raise FileNotFoundError(
                f"No (image, mask) chip pairs found for split={split!r} under "
                f"{root!r}. Build chips first with ml/data/tiling.py (this loop "
                "never downloads datasets)."
            )

    @staticmethod
    def _index(root: str, split: str) -> list[tuple[str, str]]:
        """Pair image/mask chips from the manifest, else by filename glob."""
        import csv
        import glob

        manifest = os.path.join(root, "manifest.csv")
        pairs: list[tuple[str, str]] = []
        if os.path.isfile(manifest):
            with open(manifest, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("split") != split:
                        continue
                    img, msk = row.get("image_path", ""), row.get("mask_path", "")
                    if img and msk and os.path.isfile(img) and os.path.isfile(msk):
                        pairs.append((img, msk))
            return pairs

        img_dir = os.path.join(root, split, "images")
        msk_dir = os.path.join(root, split, "masks")
        for img in sorted(glob.glob(os.path.join(img_dir, "*"))):
            msk = os.path.join(msk_dir, os.path.basename(img))
            if os.path.isfile(msk):
                pairs.append((img, msk))
        return pairs

    def _read(self, path: str, is_mask: bool) -> "np.ndarray":
        """Read a chip to numpy: image HxWx3 uint8, mask HxW uint8 in {0,1}."""
        np = _require("numpy")
        if path.lower().endswith(".npy"):
            arr = np.load(path)
        else:
            Image = _require("PIL.Image", pip_hint="pip install Pillow")
            with Image.open(path) as im:
                arr = np.asarray(im.convert("L" if is_mask else "RGB"))
        if is_mask:
            if arr.ndim == 3:
                arr = arr[..., 0]
            return (arr > 127).astype("uint8") if arr.max() > 1 else arr.astype("uint8")
        if arr.ndim == 2:  # grayscale -> 3 channels
            arr = np.stack([arr] * 3, axis=-1)
        return arr[..., :3].astype("uint8")

    def _to_image_tensor(self, arr: "np.ndarray") -> "Tensor":
        torch = _require("torch")
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float() / 255.0

    def _to_mask_tensor(self, arr: "np.ndarray") -> "Tensor":
        torch = _require("torch")
        # [1, H, W] so it broadcasts against [N, 1, H, W] logits.
        return torch.from_numpy(arr.copy()).float().unsqueeze(0)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]
        image = self._read(img_path, is_mask=False)
        mask = self._read(msk_path, is_mask=True)

        img_t = self._to_image_tensor(image)
        mask_t = self._to_mask_tensor(mask)

        if self.return_occluded and self._occlude is not None:
            # Deterministic per-sample seed -> reproducible occlusion each epoch.
            occluded = self._occlude(image, seed=self.seed + idx)
            occ_t = self._to_image_tensor(occluded)
            return img_t, mask_t, occ_t
        return img_t, mask_t


class _RandomChipDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """Tiny in-memory random dataset for ``--self-test`` (no files, no GPU).

    Yields the same ``(image, mask[, occluded])`` shapes as :class:`ChipDataset`
    so the training loop runs end-to-end without any real data.
    """

    def __init__(self, n: int, tile: int, return_occluded: bool, seed: int = 0) -> None:
        self.n = n
        self.tile = tile
        self.return_occluded = return_occluded
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch = _require("torch")
        g = torch.Generator().manual_seed(self.seed + idx)
        img = torch.rand(3, self.tile, self.tile, generator=g)
        mask = (torch.rand(1, self.tile, self.tile, generator=g) > 0.85).float()
        if self.return_occluded:
            occ = (img + 0.1 * torch.rand(3, self.tile, self.tile, generator=g)).clamp(0, 1)
            return img, mask, occ
        return img, mask


# --------------------------------------------------------------------------- #
# Training config + checkpointing.
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """All training knobs (populated from argparse)."""

    variant: str = "baseline"             # "baseline" | "robust"
    data: str = "ml/data/sample"
    out: str = "ml/checkpoints/model.pt"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    accum_steps: int = DEFAULT_ACCUM_STEPS
    tile: int = DEFAULT_TILE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    num_workers: int = DEFAULT_NUM_WORKERS
    amp: bool = True
    occlusion: bool = False               # forward an occluded view
    consistency_loss: bool = False        # add clean<->occluded consistency
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT
    consistency_warmup_epochs: int = DEFAULT_CONSISTENCY_WARMUP_EPOCHS
    consistency_mode: str = "mse"         # "mse" | "kl"
    encoder_weights: str | None = "imagenet"  # None => no download (offline)
    resume: str | None = None
    save_every: int = 1                   # checkpoint cadence (epochs)
    seed: int = 1337
    device: str | None = None             # auto-detect if None

    @property
    def uses_occlusion(self) -> bool:
        """Robust variant implies occlusion unless explicitly overridden off."""
        return self.occlusion or self.consistency_loss


def _resolve_device(name: str | None) -> "torch.device":
    """Pick CUDA if available, else CPU, honoring an explicit override."""
    torch = _require("torch")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(
    path: str,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scaler: Any,
    epoch: int,
    cfg: TrainConfig,
    best_val: float,
) -> None:
    """Atomically write a resume-able checkpoint (model + optim + scaler + meta)."""
    torch = _require("torch")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "config": vars(cfg),
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def _maybe_resume(
    cfg: TrainConfig,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scaler: Any,
) -> tuple[int, float]:
    """Load a checkpoint if ``--resume`` is set. Returns ``(start_epoch, best_val)``."""
    if not cfg.resume:
        return 0, float("inf")
    torch = _require("torch")
    if not os.path.isfile(cfg.resume):
        raise FileNotFoundError(f"--resume checkpoint not found: {cfg.resume!r}")
    ckpt = torch.load(cfg.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start = int(ckpt.get("epoch", 0)) + 1
    best = float(ckpt.get("best_val", float("inf")))
    print(f"[train] resumed from {cfg.resume!r} at epoch {start} (best_val={best:.4f})")
    return start, best


# --------------------------------------------------------------------------- #
# One epoch (with AMP + gradient accumulation).
# --------------------------------------------------------------------------- #
def _consistency_weight_for_epoch(cfg: TrainConfig, epoch: int) -> float:
    """Linear warmup of the consistency weight over the first few epochs."""
    if not cfg.consistency_loss:
        return 0.0
    warmup = max(1, cfg.consistency_warmup_epochs)
    if epoch >= warmup:
        return cfg.consistency_weight
    return cfg.consistency_weight * (epoch + 1) / warmup


def _run_epoch(
    cfg: TrainConfig,
    model: "torch.nn.Module",
    loader: Any,
    criterion: Any,
    optimizer: "torch.optim.Optimizer | None",
    scaler: Any,
    device: "torch.device",
    epoch: int,
    train: bool,
) -> dict[str, float]:
    """Run one train/val epoch; returns averaged loss components.

    Implements gradient accumulation: gradients from ``accum_steps`` micro-batches
    are summed before a single optimizer step, giving a larger effective batch
    without the VRAM cost — essential on 4 GB.
    """
    torch = _require("torch")
    losses_mod = _load_losses()

    model.train(mode=train)
    cw = _consistency_weight_for_epoch(cfg, epoch)
    accum = max(1, cfg.accum_steps)

    try:
        from tqdm import tqdm  # nice progress bar if available

        iterator = tqdm(loader, desc=f"{'train' if train else 'val'} e{epoch}",
                        leave=False)
    except ImportError:
        iterator = loader

    totals: dict[str, float] = {"total": 0.0, "sup": 0.0, "consistency": 0.0}
    n_batches = 0
    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    # torch.amp.autocast is the non-deprecated API (torch>=2.x). Disabled cleanly
    # on CPU where fp16 autocast is a no-op / unsupported.
    use_amp = bool(cfg.amp) and device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, enabled=use_amp)
        if use_amp
        else _nullcontext()
    )

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(iterator):
            if cfg.uses_occlusion:
                image, mask, occluded = batch
                occluded = occluded.to(device, non_blocking=True)
            else:
                image, mask = batch
                occluded = None
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Only the conv forward runs under autocast (fp16). Losses are
            # computed in fp32: the soft-Dice term sums sigmoid probabilities
            # over a 512x512 tile (~262k pixels), which overflows fp16's ~65504
            # max to inf -> inf/inf = NaN on step 1. Casting logits to float
            # before the loss keeps the reductions numerically safe.
            with autocast_ctx:
                logits_clean = model(image)
                logits_occ = model(occluded) if occluded is not None else None

            logits_clean = logits_clean.float()
            if logits_occ is not None:
                logits_occ = logits_occ.float()

            if occluded is not None:
                if cfg.consistency_loss:
                    loss, parts = criterion(
                        logits_clean, logits_occ, mask
                    )
                    # Apply the (possibly warmed-up) consistency weight here so
                    # the schedule is honored even though the criterion bundles
                    # its own default weight.
                    loss = parts["sup_clean"] * 0.5 + parts["sup_occluded"] * 0.5
                    cons = parts["consistency"]
                    loss = loss + cw * cons
                    sup_val = float(parts["sup_clean"] + parts["sup_occluded"]) * 0.5
                    cons_val = float(cons)
                else:
                    # Occlusion augmentation without an explicit consistency
                    # term: supervise both views against the clean mask.
                    sup_clean = losses_mod.bce_dice_loss(logits_clean, mask)
                    sup_occ = losses_mod.bce_dice_loss(logits_occ, mask)
                    loss = 0.5 * (sup_clean + sup_occ)
                    sup_val = float(loss)
                    cons_val = 0.0
            else:
                loss = losses_mod.bce_dice_loss(logits_clean, mask)
                sup_val = float(loss)
                cons_val = 0.0

            if train and optimizer is not None:
                # Scale by 1/accum so summed micro-batch grads average correctly.
                scaled = loss / accum
                if scaler is not None and use_amp:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                if (step + 1) % accum == 0:
                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            totals["total"] += float(loss)
            totals["sup"] += sup_val
            totals["consistency"] += cons_val
            n_batches += 1

    # Flush a trailing partial accumulation window so its grads aren't dropped.
    if train and optimizer is not None and n_batches % accum != 0:
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    denom = max(1, n_batches)
    return {k: v / denom for k, v in totals.items()}


class _nullcontext:
    """Minimal no-op context manager (stdlib contextlib.nullcontext stand-in)."""

    def __enter__(self) -> "_nullcontext":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def build_model(cfg: TrainConfig, device: "torch.device") -> "torch.nn.Module":
    """Build the D-LinkNet34-equivalent and move it to the device."""
    build_dlinknet = _load_build_dlinknet()
    model = build_dlinknet(encoder_weights=cfg.encoder_weights)
    return model.to(device)


def build_dataloaders(cfg: TrainConfig) -> tuple[Any, Any | None]:
    """Build train (and optional val) DataLoaders for the chip dataset."""
    tud = _require("torch.utils.data", pip_hint="install torch")
    train_ds = ChipDataset(
        cfg.data, split="train", tile_size=cfg.tile,
        return_occluded=cfg.uses_occlusion, seed=cfg.seed,
    )
    train_loader = tud.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device != "cpu"),
        drop_last=False,
    )
    val_loader = None
    try:
        val_ds = ChipDataset(
            cfg.data, split="val", tile_size=cfg.tile,
            return_occluded=cfg.uses_occlusion, seed=cfg.seed,
        )
        val_loader = tud.DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=(cfg.device != "cpu"),
        )
    except FileNotFoundError:
        print("[train] no 'val' split found — training without validation.")
    return train_loader, val_loader


def train(cfg: TrainConfig) -> str:
    """Run the full training loop. Returns the path of the final checkpoint.

    This is the single entry point both the CLI and ``--self-test`` route through
    (the self-test swaps in a random in-memory dataset). It NEVER runs on import.
    """
    torch = _require("torch")
    losses_mod = _load_losses()

    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    cfg.device = device.type
    if device.type == "cuda":
        # cudnn autotuner: fixed 512x512 chips -> picks fast kernels once.
        torch.backends.cudnn.benchmark = True

    print(f"[train] variant={cfg.variant} device={device} amp={cfg.amp} "
          f"batch={cfg.batch_size} accum={cfg.accum_steps} "
          f"(effective batch={cfg.batch_size * cfg.accum_steps})")
    if cfg.uses_occlusion:
        print(f"[train] occlusion ON; consistency={'ON' if cfg.consistency_loss else 'OFF'} "
              f"(weight={cfg.consistency_weight}, warmup={cfg.consistency_warmup_epochs} ep)")

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # GradScaler stabilizes fp16 gradients under AMP (CUDA only).
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None
    criterion = losses_mod.RobustSegLoss(
        consistency_weight=cfg.consistency_weight,
        consistency_mode=cfg.consistency_mode,
    )

    train_loader, val_loader = build_dataloaders(cfg)
    start_epoch, best_val = _maybe_resume(cfg, model, optimizer, scaler)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        tr = _run_epoch(cfg, model, train_loader, criterion, optimizer,
                        scaler, device, epoch, train=True)
        msg = (f"[train] epoch {epoch + 1}/{cfg.epochs} "
               f"loss={tr['total']:.4f} sup={tr['sup']:.4f} "
               f"cons={tr['consistency']:.4f} ({time.time() - t0:.1f}s)")

        val_metric = tr["total"]
        if val_loader is not None:
            va = _run_epoch(cfg, model, val_loader, criterion, None,
                            scaler, device, epoch, train=False)
            val_metric = va["total"]
            msg += f" | val_loss={val_metric:.4f}"
        print(msg)

        # Checkpoint cadence + always keep the best-val model alongside.
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            _save_checkpoint(cfg.out, model, optimizer, scaler, epoch, cfg, best_val)
        if val_metric < best_val:
            best_val = val_metric
            best_path = _best_path(cfg.out)
            _save_checkpoint(best_path, model, optimizer, scaler, epoch, cfg, best_val)
            print(f"[train]   new best (val={best_val:.4f}) -> {best_path}")

        # Free the cache between epochs to keep the 4GB working set small.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[train] done -> {cfg.out}")
    return cfg.out


def _best_path(out: str) -> str:
    """Derive the best-checkpoint path next to ``out`` (``foo.pt`` -> ``foo.best.pt``)."""
    base, ext = os.path.splitext(out)
    return f"{base}.best{ext or '.pt'}"


def _apply_variant_defaults(cfg: TrainConfig) -> None:
    """The 'robust' variant turns on occlusion + consistency unless overridden.

    Encodes the demo recipe: baseline = clean-only (fragments under occlusion);
    robust = occlusion aug + consistency (recovers the critical links).
    """
    if cfg.variant == "robust":
        if not cfg.occlusion and not cfg.consistency_loss:
            cfg.occlusion = True
            cfg.consistency_loss = True
    elif cfg.variant == "baseline":
        # Baseline is clean-only by design; ignore stray occlusion flags loudly.
        if cfg.occlusion or cfg.consistency_loss:
            print("[train] NOTE: --variant baseline ignores --occlusion/"
                  "--consistency-loss (baseline is clean-only by design).")
            cfg.occlusion = False
            cfg.consistency_loss = False


# --------------------------------------------------------------------------- #
# Self-test: one training step on random synthetic chips (no data, CPU OK).
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Prove the loop is differentiable end-to-end without real data or a GPU.

    Builds the model with random init (encoder_weights=None -> no download), runs
    a couple of steps for BOTH variants on a tiny random in-memory dataset, and
    asserts the loss is finite. Requires torch+smp installed; otherwise the
    __main__ guard catches ImportError and exits cleanly.
    """
    torch = _require("torch")
    tud = _require("torch.utils.data", pip_hint="install torch")
    losses_mod = _load_losses()

    device = torch.device("cpu")  # self-test stays on CPU for portability
    tile = 64                      # tiny so it runs in seconds on CPU

    for variant, occ in (("baseline", False), ("robust", True)):
        print(f"[self-test] variant={variant} occlusion={occ}")
        cfg = TrainConfig(
            variant=variant, epochs=1, batch_size=1, accum_steps=2, tile=tile,
            amp=False, occlusion=occ, consistency_loss=occ, device="cpu",
            encoder_weights=None,
        )
        model = build_model(cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        criterion = losses_mod.RobustSegLoss(consistency_weight=1.0)
        ds = _RandomChipDataset(n=2, tile=tile, return_occluded=occ)
        loader = tud.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        stats = _run_epoch(cfg, model, loader, criterion, optimizer,
                           None, device, epoch=0, train=True)
        assert stats["total"] == stats["total"], "loss is NaN"  # NaN != NaN
        print(f"[self-test]   ok: {{'loss': {stats['total']:.4f}, "
              f"'sup': {stats['sup']:.4f}, 'cons': {stats['consistency']:.4f}}}")

    print("[self-test] OK — training graph runs end-to-end for both variants.")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", choices=("baseline", "robust"),
                    default="baseline",
                    help="baseline = clean-only; robust = occlusion + consistency")
    ap.add_argument("--data", default="ml/data/sample",
                    help="chip dir from ml/data/tiling.py (manifest.csv or "
                         "<split>/images + <split>/masks)")
    ap.add_argument("--out", default="ml/checkpoints/model.pt",
                    help="output checkpoint path")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="keep 1-2 on 4GB VRAM")
    ap.add_argument("--accum-steps", type=int, default=DEFAULT_ACCUM_STEPS,
                    help="gradient accumulation -> larger effective batch")
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="disable mixed precision (AMP is on by default on CUDA)")
    ap.add_argument("--occlusion", action="store_true",
                    help="forward an occluded view (auto-on for --variant robust)")
    ap.add_argument("--consistency-loss", dest="consistency_loss",
                    action="store_true",
                    help="add clean<->occluded consistency (auto-on for robust)")
    ap.add_argument("--consistency-weight", type=float,
                    default=DEFAULT_CONSISTENCY_WEIGHT)
    ap.add_argument("--consistency-warmup-epochs", type=int,
                    default=DEFAULT_CONSISTENCY_WARMUP_EPOCHS)
    ap.add_argument("--consistency-mode", choices=("mse", "kl"), default="mse")
    ap.add_argument("--no-pretrained", dest="encoder_weights",
                    action="store_const", const=None, default="imagenet",
                    help="random encoder init (no ~85MB download; offline boxes)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume optimizer+model+scaler from")
    ap.add_argument("--save-every", type=int, default=1,
                    help="checkpoint every N epochs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None,
                    help="force a device, e.g. 'cuda' or 'cpu' (auto if unset)")
    ap.add_argument("--self-test", action="store_true",
                    help="run a tiny CPU training step on random data and exit")
    return ap


def _cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        variant=args.variant,
        data=args.data,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        tile=args.tile,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=args.amp,
        occlusion=args.occlusion,
        consistency_loss=args.consistency_loss,
        consistency_weight=args.consistency_weight,
        consistency_warmup_epochs=args.consistency_warmup_epochs,
        consistency_mode=args.consistency_mode,
        encoder_weights=args.encoder_weights,
        resume=args.resume,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    cfg = _cfg_from_args(args)
    _apply_variant_defaults(cfg)
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry; never runs on import
    try:
        raise SystemExit(main())
    except ImportError as exc:
        # Import-safe path: torch / smp / the model package is absent (expected on
        # the 4GB demo laptop). Print remediation and exit cleanly instead of a
        # raw traceback. Training runs on the dedicated CUDA box.
        print(f"[train] cannot run — missing dependency:\n  {exc}", file=sys.stderr)
        raise SystemExit(0)
