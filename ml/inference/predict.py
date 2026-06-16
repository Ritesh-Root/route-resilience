"""Checkpoint inference — predict a binary road mask from a satellite image.

Role in the project
-------------------
This is the function the **backend segmenter hook** calls to turn a satellite
RGB image into a single-channel binary road mask (1 = road, 0 = background).
The mask is then vectorized/graphed downstream to build the road network that
the FastAPI ``/api/network`` endpoint serves.

It loads a trained checkpoint of the D-LinkNet-equivalent model built by
``ml/models/dlinknet.py`` (LinkNet + ResNet34 encoder, logits out) and runs:

  * a single 512x512 forward pass for a tile, OR
  * an overlap-blended **sliding window** for any larger image, so memory stays
    bounded on the 4GB RTX 3050 (we never feed a full 1024+ tile at once).

Two checkpoints exist in the demo story (same architecture, different weights):
the ``baseline`` model FRAGMENTS the graph under occlusion, while the ``robust``
model — trained with the synthetic occlusion augmentation from
``backend/app/services/occlusion.py`` (clouds / shadows / canopy) plus a
consistency loss — recovers the critical links the baseline loses. This file is
weights-agnostic: point ``load_model`` at whichever ``.pt``/``.pth`` you trained.

Import safety
-------------
Every heavy dependency (torch, numpy, PIL, rasterio, segmentation-models-pytorch)
is imported lazily inside functions via :func:`_require`, so this module imports
cleanly on a machine with none of them installed (the Python-3.14 dev box). The
heavy work only happens when you actually call ``load_model`` / ``predict_*``.

HARD CONSTRAINTS (Python 3.14, 4GB RTX 3050 laptop GPU)
-------------------------------------------------------
  * Do NOT train here and do NOT download datasets — this is inference only.
  * Keep ``window=512`` (and small overlap) so a single forward pass fits 4GB.
  * torch / smp may lack cp314 wheels; if so, run inference in a py3.12 env
    (see ml/requirements-ml.txt). Installing those libs is a separate step.

Install (see ml/requirements-ml.txt for the full pinned stack + caveats):
    pip install torch torchvision segmentation-models-pytorch numpy pillow
    # rasterio is OPTIONAL (only needed to read GeoTIFF inputs).

Usage
-----
    from ml.inference.predict import load_model, predict_image

    model = load_model("ml/checkpoints/robust.pt", device="cuda")
    mask = predict_image(model, "ml/data/sample/tile.png")  # -> uint8 HxW {0,1}

CLI smoke test (no weights / no torch required to *import*; see __main__):
    python -m ml.inference.predict --self-test
    python -m ml.inference.predict --checkpoint ml/checkpoints/robust.pt \
        --image "ml/data/sample/tile.png" --out mask.png
"""
from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

# Type-only imports: evaluated by type checkers, never at runtime, so importing
# this module never requires numpy/torch to be installed.
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import torch
    import torch.nn as nn


# --- Inference defaults ------------------------------------------------------
DEFAULT_WINDOW: int = 512          # tile size the model was trained at
DEFAULT_OVERLAP: int = 64          # px overlap between windows (seam blending)
DEFAULT_THRESHOLD: float = 0.5     # sigmoid probability cutoff for "road"
# ImageNet normalization (matches the smp ResNet34 encoder's pretrained weights;
# see ml/models/dlinknet.get_preprocessing_params for the authoritative source).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _require(module: str, pip_name: str | None = None) -> Any:
    """Import ``module`` lazily, raising a clear, actionable error if missing.

    Keeps this file import-safe on machines without torch/numpy/PIL installed
    (mirrors the helper in ``ml/data/deepglobe.py``).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        pkg = pip_name or module
        raise ImportError(
            f"ml.inference.predict needs '{module}'. Install it with "
            f"`pip install {pkg}`. See ml/requirements-ml.txt for pinned "
            f"versions and the Python-3.14 / GDAL caveats (torch/smp may need a "
            f"py3.12 env)."
        ) from exc


def _load_model_builder():
    """Return ``build_dlinknet`` from ``ml/models/dlinknet.py``.

    Imported by file path so this works whether or not ``ml`` is on ``sys.path``
    (the backend may import this module from an arbitrary CWD). Falls back to the
    normal package import if the path lookup is unavailable.
    """
    import importlib.util

    # ml/inference/predict.py -> ml/ is two parents up; models/dlinknet.py beside it.
    here = os.path.dirname(os.path.abspath(__file__))
    ml_root = os.path.abspath(os.path.join(here, ".."))
    dlink_path = os.path.join(ml_root, "models", "dlinknet.py")

    if os.path.isfile(dlink_path):
        spec = importlib.util.spec_from_file_location("_rr_dlinknet", dlink_path)
        assert spec and spec.loader  # for type checkers
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_dlinknet

    # Fallback: rely on package resolution if the file isn't where we expect.
    try:
        from ml.models.dlinknet import build_dlinknet  # type: ignore
        return build_dlinknet
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            f"Could not locate the model builder at {dlink_path!r} nor import "
            f"ml.models.dlinknet. Ensure ml/models/dlinknet.py exists."
        ) from exc


def _pick_device(device: str | None) -> str:
    """Resolve the inference device, preferring CUDA when available.

    Returns a device string ("cuda" / "cpu"). Honors an explicit request but
    falls back to CPU with no error if CUDA is unavailable.
    """
    torch = _require("torch")
    if device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    checkpoint_path: str,
    device: str | None = None,
    encoder_name: str = "resnet34",
    *,
    strict: bool = True,
) -> "nn.Module":
    """Build the D-LinkNet-equivalent model and load trained weights.

    The checkpoint may be either a raw ``state_dict`` or a training dict that
    nests it under ``"state_dict"`` / ``"model"`` (common Lightning/torch styles).
    The model is built with ``encoder_weights=None`` so loading triggers **no**
    ImageNet download — the trained weights fully replace the backbone.

    Args:
        checkpoint_path: path to a ``.pt`` / ``.pth`` file. Quote paths with spaces.
        device: "cuda", "cpu", or None to auto-pick (CUDA if available).
        encoder_name: must match the architecture the checkpoint was trained with.
        strict: forwarded to ``load_state_dict``; set False to tolerate minor
            key mismatches (e.g. a wrapped/compiled model prefix).

    Returns:
        An ``nn.Module`` in ``.eval()`` mode on the chosen device.

    Raises:
        FileNotFoundError: if ``checkpoint_path`` does not exist.
        ImportError: if torch / segmentation-models-pytorch are not installed.
    """
    torch = _require("torch")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path!r}. Train one under ml/train/ "
            f"on a CUDA machine (do NOT train on the 4GB dev box), or point this "
            f"at an existing baseline/robust .pt file."
        )

    dev = _pick_device(device)
    build_dlinknet = _load_model_builder()
    # encoder_weights=None -> no internet/download; trained weights overwrite all.
    model = build_dlinknet(encoder_name=encoder_name, encoder_weights=None)

    ckpt = torch.load(checkpoint_path, map_location=dev)
    # Unwrap common training-checkpoint containers down to a bare state_dict.
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    # Strip a leading "module." (DataParallel) or "model." (Lightning) prefix.
    if isinstance(ckpt, dict):
        ckpt = {
            k.removeprefix("module.").removeprefix("model."): v
            for k, v in ckpt.items()
        }

    model.load_state_dict(ckpt, strict=strict)
    model.to(dev).eval()
    return model


def _read_image(image: "str | np.ndarray") -> "np.ndarray":
    """Load an image as an HxWx3 uint8 RGB numpy array.

    Accepts a numpy array (returned as-is after RGB/contiguity checks) or a path
    to a PNG/JPG (via PIL) or a GeoTIFF (via rasterio, imported only if needed).
    """
    np = _require("numpy")

    if not isinstance(image, str):
        arr = np.asarray(image)
        if arr.ndim == 2:  # grayscale -> 3-channel
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:  # drop alpha
            arr = arr[..., :3]
        return np.ascontiguousarray(arr)

    ext = os.path.splitext(image)[1].lower()
    if ext in (".tif", ".tiff"):
        rasterio = _require("rasterio")  # OPTIONAL: only GeoTIFF inputs need it
        with rasterio.open(image) as src:
            # rasterio is band-first (C, H, W); take first 3 bands -> HWC.
            data = src.read()[:3]
        return np.ascontiguousarray(np.transpose(data, (1, 2, 0)))

    Image = _require("PIL.Image", pip_name="Pillow")
    with Image.open(image) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def _normalize_chip(chip: "np.ndarray") -> "torch.Tensor":
    """HWC uint8 chip -> normalized [1, 3, H, W] float tensor (ImageNet stats)."""
    torch = _require("torch")
    np = _require("numpy")
    x = chip.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    x = (x - mean) / std
    t = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy()).unsqueeze(0)
    return t


def predict_tile(
    model: "nn.Module",
    tile: "np.ndarray",
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
) -> "np.ndarray":
    """Predict the road mask for a single chip (no windowing).

    Args:
        model: a model from :func:`load_model`.
        tile: an HxWx3 uint8 RGB array (ideally the model's training size, 512).
        threshold: sigmoid probability cutoff for the road class.
        device: device override; defaults to the model's current device.

    Returns:
        An HxW ``uint8`` numpy mask with values in {0, 1}.
    """
    torch = _require("torch")
    np = _require("numpy")

    dev = device or next(model.parameters()).device
    x = _normalize_chip(tile).to(dev)
    model.eval()
    with torch.no_grad():
        logits = model(x)                 # [1, 1, H, W]
        probs = torch.sigmoid(logits)[0, 0]
        mask = (probs > threshold).to(torch.uint8)
    return mask.detach().cpu().numpy().astype(np.uint8)


def predict_sliding_window(
    model: "nn.Module",
    image: "np.ndarray",
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
) -> "np.ndarray":
    """Predict a road mask for an arbitrarily large image via overlap blending.

    The image is tiled into ``window x window`` chips that step by ``window -
    overlap``; per-pixel sigmoid probabilities are averaged in the overlap
    regions (soft seam blending) before thresholding once at the end. This keeps
    peak VRAM at one chip's worth, so large tiles fit the 4GB GPU.

    Edge windows are clamped to the image bounds (so the last row/column is
    always covered even when the image size isn't a multiple of the step).

    Args:
        model: a model from :func:`load_model`.
        image: an HxWx3 uint8 RGB array of any size >= 1px.
        window: chip size fed to the model (keep at the training size, 512).
        overlap: pixels of overlap between adjacent chips (0 <= overlap < window).
        threshold: final sigmoid cutoff applied to the blended probability map.
        device: device override; defaults to the model's current device.

    Returns:
        An HxW ``uint8`` numpy mask with values in {0, 1}, same H/W as ``image``.
    """
    torch = _require("torch")
    np = _require("numpy")

    if not (0 <= overlap < window):
        raise ValueError(f"overlap must satisfy 0 <= overlap < window={window}")

    h, w = image.shape[:2]
    # Small images: a single (clamped) forward pass is enough.
    if h <= window and w <= window:
        return predict_tile(model, image, threshold=threshold, device=device)

    dev = device or next(model.parameters()).device
    step = window - overlap
    # Accumulate summed probabilities + a count map, then average and threshold.
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    def _starts(extent: int) -> list[int]:
        # Window start positions; ensure the final window touches the edge.
        if extent <= window:
            return [0]
        xs = list(range(0, extent - window + 1, step))
        if xs[-1] != extent - window:
            xs.append(extent - window)
        return xs

    ys, xs = _starts(h), _starts(w)
    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                chip = image[y0:y0 + window, x0:x0 + window]
                x = _normalize_chip(chip).to(dev)
                probs = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
                ph, pw = chip.shape[:2]
                prob_sum[y0:y0 + ph, x0:x0 + pw] += probs[:ph, :pw]
                count[y0:y0 + ph, x0:x0 + pw] += 1.0

    count[count == 0] = 1.0  # guard (shouldn't happen given edge clamping)
    avg = prob_sum / count
    return (avg > threshold).astype(np.uint8)


def predict_image(
    model: "nn.Module",
    image: "str | np.ndarray",
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
) -> "np.ndarray":
    """High-level entry point the backend segmenter hook calls.

    Loads ``image`` (path or array; PNG/JPG via PIL, GeoTIFF via rasterio),
    auto-selects a single forward pass (<= one window) or the sliding-window
    path (larger), and returns the binary road mask.

    Args:
        model: a model from :func:`load_model`.
        image: image path (str) or an HxWx3 / HxW numpy array.
        window: model input size (training size, default 512).
        overlap: sliding-window overlap in pixels.
        threshold: sigmoid road-probability cutoff.
        device: device override; defaults to the model's current device.

    Returns:
        An HxW ``uint8`` numpy mask with values in {0, 1}.
    """
    arr = _read_image(image)
    return predict_sliding_window(
        model, arr, window=window, overlap=overlap,
        threshold=threshold, device=device,
    )


def _save_mask(mask: "np.ndarray", out_path: str) -> None:
    """Write a {0,1} mask to disk as an 8-bit PNG (white = road)."""
    Image = _require("PIL.Image", pip_name="Pillow")
    np = _require("numpy")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)


def _self_test() -> None:
    """Import-safe smoke test: never loads weights, never downloads.

    Builds the architecture with random init (no ImageNet download), runs the
    sliding-window predictor on a synthetic 768x1024 image, and reports the
    mask shape. Prints a clear, actionable message if torch/smp are absent.
    """
    try:
        np = _require("numpy")
        _require("torch")
    except ImportError as exc:
        print(f"[skip self-test] {exc}")
        return

    print("Building D-LinkNet-equivalent with random init (no download)...")
    build_dlinknet = _load_model_builder()
    model = build_dlinknet(encoder_weights=None).eval()

    # Synthetic non-square image to exercise edge-clamped sliding windows.
    rng = np.random.default_rng(0)
    fake = rng.integers(0, 256, size=(768, 1024, 3), dtype=np.uint8)
    mask = predict_sliding_window(model, fake, window=512, overlap=64)
    print(
        f"OK: input {fake.shape} -> mask {mask.shape}, dtype={mask.dtype}, "
        f"road fraction={float(mask.mean()):.3f} (random weights, so meaningless)"
    )


def main() -> None:
    """CLI: run inference on one image, or a no-weights self-test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict a binary road mask from a satellite image."
    )
    parser.add_argument("--checkpoint", help="path to trained .pt/.pth weights")
    parser.add_argument("--image", help="input image (PNG/JPG/GeoTIFF)")
    parser.add_argument("--out", default="mask.png", help="output mask PNG path")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    parser.add_argument(
        "--self-test", action="store_true",
        help="build with random weights and run a synthetic forward pass",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.checkpoint or not args.image:
        parser.error("provide --checkpoint and --image, or use --self-test")

    model = load_model(args.checkpoint, device=args.device)
    mask = predict_image(
        model, args.image,
        window=args.window, overlap=args.overlap, threshold=args.threshold,
    )
    _save_mask(mask, args.out)
    print(f"Wrote mask {mask.shape} -> {args.out}")


if __name__ == "__main__":
    main()
