"""DeepGlobe Road Extraction dataset — torch ``Dataset`` (RGB tile + binary mask).

This module provides :class:`DeepGlobeRoadDataset`, a PyTorch ``Dataset`` that
yields ``(image, mask)`` pairs for the road-segmentation model. Each sample is an
RGB satellite tile and a single-channel binary road mask (1 = road, 0 = background).

For the robustness INNOVATION we optionally return a *clean* and an *occluded*
copy of the same tile (sharing one ground-truth mask). The occluded copy is made
with the numpy-only synthetic occlusion generators in
``backend/app/services/occlusion.py`` (clouds / shadows / canopy). Train the network
to predict the true mask from the occluded image — optionally adding a consistency
loss between the clean and occluded predictions — so it learns to "see through"
occlusion. See ``ml/train/`` (per PROJECT-PLAN) for the training loop.

------------------------------------------------------------------------------
DOWNLOAD (do this manually — this file NEVER downloads anything)
------------------------------------------------------------------------------
Dataset: Kaggle ``balraj98/deepglobe-road-extraction-dataset`` (~6 GB).
WARNING: large. On the 4GB RTX 3050 / Python 3.14 dev box, prefer a tiny subset.

  # 1) Install the Kaggle CLI and place your token at ~/.kaggle/kaggle.json
  pip install kaggle
  # 2) Download + unzip (run from the project root; quote paths with spaces):
  kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset \
      -p "atlas-vision/ml/data/deepglobe" --unzip

Expected layout after unzip (DeepGlobe ships only ``train/`` with masks):

  atlas-vision/ml/data/deepglobe/
    train/
      <id>_sat.jpg     # RGB satellite tile (1024x1024)
      <id>_mask.png    # binary road mask (white = road)
    valid/             # images only, NO masks (test split — unusable for training)
    test/              # images only, NO masks
    metadata.csv

Only ``train/`` has masks, so we split that folder ourselves (see ``split=``).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    from ml.data.deepglobe import DeepGlobeRoadDataset
    ds = DeepGlobeRoadDataset(
        root="atlas-vision/ml/data/deepglobe",
        split="train",
        tile_size=512,
        return_occluded=True,   # enables the robustness pair
    )
    img, mask, occ = ds[0]      # occ present only when return_occluded=True

Heavy deps (torch, PIL/opencv, albumentations) are imported lazily/guarded so this
file is import-safe even when those libs are absent (Python 3.14 box without torch).

TODO(train): wire this into ml/train/train.py with a DataLoader and the U-Net/DLinkNet
model; keep batch size small (e.g. 2-4 at tile_size<=512) to fit 4 GB VRAM.
"""
from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING, Any

# Type-only imports: evaluated by type checkers, never at runtime, so importing
# this module does not require numpy/torch to be installed.
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from torch import Tensor
    from torch.utils.data import Dataset as _TorchDataset
else:
    # At runtime we don't have a hard base class — define a no-op stand-in so the
    # module imports even without torch. The real torch.Dataset is mixed in lazily
    # via ``_torch_dataset_base`` only when actually instantiated/used.
    _TorchDataset = object


_SAT_SUFFIX = "_sat.jpg"
_MASK_SUFFIX = "_mask.png"


def _require(module: str, pip_name: str | None = None) -> Any:
    """Import ``module`` lazily, raising a clear, actionable error if it's missing.

    Keeps this file import-safe on machines without torch/numpy/PIL installed.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on env
        pkg = pip_name or module
        raise ImportError(
            f"DeepGlobeRoadDataset needs '{module}'. Install it with "
            f"`pip install {pkg}` (note: do NOT pip install torch on the 4GB "
            f"dev box — train on a CUDA machine)."
        ) from exc


def _load_occluder():
    """Return ``random_occlude`` from the backend occlusion service.

    The occlusion generators live in ``backend/app/services/occlusion.py`` and are
    numpy-only. We import by file path so this works whether or not ``backend`` is
    on ``sys.path``. Falls back to identity (with a warning) if unavailable.
    """
    import importlib.util
    import warnings

    # ml/data/deepglobe.py -> project root is three parents up from this file,
    # then backend/app/services/occlusion.py.
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    occ_path = os.path.join(
        project_root, "backend", "app", "services", "occlusion.py"
    )
    if not os.path.isfile(occ_path):
        warnings.warn(
            f"occlusion.py not found at {occ_path!r}; occluded view = clean view.",
            RuntimeWarning,
            stacklevel=2,
        )
        return lambda img, seed=None: img

    spec = importlib.util.spec_from_file_location("_rr_occlusion", occ_path)
    assert spec and spec.loader  # for type checkers
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.random_occlude


def _torch_dataset_base():
    """Return ``torch.utils.data.Dataset`` lazily (raises a clear error if absent)."""
    tud = _require("torch.utils.data", pip_name="torch")
    return tud.Dataset


class DeepGlobeRoadDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """RGB tile + binary road mask pairs from the DeepGlobe Road Extraction set.

    Only the official ``train/`` folder has masks, so we deterministically split it
    into our own train/val partitions (by sorted filename + a fixed seed).

    Parameters
    ----------
    root:
        Path to the unzipped dataset (the folder containing ``train/``).
    split:
        ``"train"`` or ``"val"`` — both drawn from the official ``train/`` folder.
    tile_size:
        Square size to which tiles/masks are resized (default 512). Keep small on
        4 GB VRAM. ``None`` keeps the native 1024x1024 resolution.
    val_fraction:
        Fraction of the official train folder held out for validation.
    seed:
        Seed for the deterministic train/val split and per-sample occlusion.
    return_occluded:
        If ``True``, ``__getitem__`` returns ``(image, mask, occluded)``; the
        occluded view is produced by the backend's ``random_occlude``. Otherwise
        returns ``(image, mask)``.
    transform:
        Optional callable applied to the numpy image/mask BEFORE tensor conversion,
        e.g. an albumentations ``Compose``. It must accept ``image=`` and ``mask=``
        keyword args and return a dict with ``"image"`` and ``"mask"`` keys.
    as_tensor:
        If ``True`` (default) convert outputs to ``torch.Tensor`` (CHW float image
        in [0,1], HxW float mask in {0,1}). If ``False`` return numpy arrays
        (HWC uint8 image, HxW uint8 mask) — handy for quick inspection without torch.

    Notes
    -----
    Heavy work (decoding images, torch conversion) happens only inside
    ``__getitem__`` / ``__len__`` so merely importing or even constructing the
    dataset stays lightweight and dependency-guarded.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        tile_size: int | None = 512,
        val_fraction: float = 0.1,
        seed: int = 1337,
        return_occluded: bool = False,
        transform: Any | None = None,
        as_tensor: bool = True,
    ) -> None:
        # Mix in the real torch Dataset base when available so DataLoader treats
        # instances correctly; harmless no-op if torch is missing at import time.
        try:
            base = _torch_dataset_base()
            if base is not object and not isinstance(self, base):
                # Re-parent the class once, lazily, on first construction.
                cls = type(self)
                if base not in cls.__bases__:
                    cls.__bases__ = (base,)
        except ImportError:
            # torch absent: dataset still usable with as_tensor=False for inspection.
            pass

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

        self.root = root
        self.split = split
        self.tile_size = tile_size
        self.seed = seed
        self.return_occluded = return_occluded
        self.transform = transform
        self.as_tensor = as_tensor

        self._occlude = _load_occluder() if return_occluded else None
        self.pairs: list[tuple[str, str]] = self._index(root, split, val_fraction, seed)

    # ------------------------------------------------------------------ indexing
    @staticmethod
    def _index(
        root: str, split: str, val_fraction: float, seed: int
    ) -> list[tuple[str, str]]:
        """Discover ``(sat, mask)`` path pairs and apply a deterministic split."""
        train_dir = os.path.join(root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Expected DeepGlobe 'train/' under {root!r} (got none). See the "
                f"DOWNLOAD instructions in this file's docstring — nothing is "
                f"downloaded automatically."
            )

        sats = sorted(glob.glob(os.path.join(train_dir, f"*{_SAT_SUFFIX}")))
        pairs: list[tuple[str, str]] = []
        for sat in sats:
            mask = sat[: -len(_SAT_SUFFIX)] + _MASK_SUFFIX
            if os.path.isfile(mask):
                pairs.append((sat, mask))

        if not pairs:
            raise FileNotFoundError(
                f"No '*{_SAT_SUFFIX}' / '*{_MASK_SUFFIX}' pairs found in {train_dir!r}."
            )

        # Deterministic shuffle (stdlib only — no numpy needed just to split).
        import random

        rng = random.Random(seed)
        order = list(range(len(pairs)))
        rng.shuffle(order)
        pairs = [pairs[i] for i in order]

        n_val = max(1, int(round(len(pairs) * val_fraction)))
        return pairs[n_val:] if split == "train" else pairs[:n_val]

    # -------------------------------------------------------------------- helpers
    def _read_image(self, path: str) -> "np.ndarray":
        """Read an RGB image as an HxWx3 uint8 numpy array (lazy PIL import)."""
        np = _require("numpy")
        Image = _require("PIL.Image", pip_name="Pillow")
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        return self._resize(arr, is_mask=False)

    def _read_mask(self, path: str) -> "np.ndarray":
        """Read a binary mask as an HxW uint8 array in {0,1} (lazy PIL import)."""
        np = _require("numpy")
        Image = _require("PIL.Image", pip_name="Pillow")
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"))
        arr = (arr > 127).astype("uint8")  # white road -> 1
        return self._resize(arr, is_mask=True)

    def _resize(self, arr: "np.ndarray", is_mask: bool) -> "np.ndarray":
        """Resize to ``self.tile_size`` (nearest for masks, bilinear for images)."""
        if self.tile_size is None:
            return arr
        np = _require("numpy")
        Image = _require("PIL.Image", pip_name="Pillow")
        size = (self.tile_size, self.tile_size)
        if is_mask:
            im = Image.fromarray(arr * 255).resize(size, Image.NEAREST)
            return (np.asarray(im) > 127).astype("uint8")
        im = Image.fromarray(arr).resize(size, Image.BILINEAR)
        return np.asarray(im)

    def _to_tensors(self, *arrays: "np.ndarray"):
        """Convert numpy arrays to torch tensors (image CHW float, mask HxW float)."""
        torch = _require("torch")
        out = []
        for a in arrays:
            if a.ndim == 3:  # HWC image -> CHW float in [0,1]
                t = torch.from_numpy(a.transpose(2, 0, 1).copy()).float() / 255.0
            else:  # HxW mask -> float {0,1}
                t = torch.from_numpy(a.copy()).float()
            out.append(t)
        return tuple(out)

    # ----------------------------------------------------------- Dataset protocol
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Return ``(image, mask)`` or ``(image, mask, occluded)`` for sample ``idx``.

        With ``as_tensor=True`` (default) tensors are returned; otherwise numpy.
        """
        sat_path, mask_path = self.pairs[idx]
        image = self._read_image(sat_path)
        mask = self._read_mask(mask_path)

        # Optional albumentations-style transform on the clean numpy pair.
        if self.transform is not None:
            res = self.transform(image=image, mask=mask)
            image, mask = res["image"], res["mask"]

        occluded = None
        if self.return_occluded and self._occlude is not None:
            # Per-sample deterministic seed -> reproducible occlusion across epochs.
            occluded = self._occlude(image, seed=self.seed + idx)

        if not self.as_tensor:
            return (image, mask, occluded) if occluded is not None else (image, mask)

        if occluded is not None:
            img_t, mask_t, occ_t = self._to_tensors(image, mask, occluded)
            return img_t, mask_t, occ_t
        img_t, mask_t = self._to_tensors(image, mask)
        return img_t, mask_t


def _selftest() -> None:
    """Import-safe smoke test: never downloads, never requires torch.

    Verifies the module imports and that indexing fails loudly (not silently) when
    the dataset is absent — which is the expected state on the dev box.
    """
    print("[deepglobe] module imported OK (no heavy deps required).")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepglobe")
    print(f"[deepglobe] looking for dataset under: {root}")
    if not os.path.isdir(os.path.join(root, "train")):
        print(
            "[deepglobe] dataset not present (expected). See the DOWNLOAD section "
            "in this file's docstring to fetch it manually on a training machine."
        )
        return
    ds = DeepGlobeRoadDataset(root=root, split="train", as_tensor=False)
    print(f"[deepglobe] indexed {len(ds)} train pairs; first sample shapes:")
    sample = ds[0]
    for i, part in enumerate(sample):
        shape = getattr(part, "shape", None)
        print(f"  [{i}] shape={shape}")


if __name__ == "__main__":
    _selftest()
