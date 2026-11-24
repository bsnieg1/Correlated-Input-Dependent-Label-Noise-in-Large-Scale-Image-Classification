"""ANIMAL-10N (Song, Kim, Lee, ICML 2019): 10 animal classes in 5 confusable pairs, 64x64 RGB.

50 000 training images with real human label noise (~8%, unknown which ones) and 5 000 clean test images.
Two distributions by KAIST are supported:
    raw image version: training/ and testing/ folders of "<label>_<name>.jpg" (raw_image_ver.zip, MD5
        f0f2b420c3371b516728ed0ef9fb2108; the link to it is quoted in several public noisy-label repositories);
    binary version: data_batch_*.bin, test_batch.bin; each record is
        <4-byte id><4-byte label><3 x 64 x 64 uint8 image, channel-major>.
Both are cached as one uint8 tensor file (animal10n.pt).

Splits are (x uint8 [N, 3, 64, 64], y_given, y_reference). With no injected noise y_reference = y_given (the
original labels; their own ~8% noise is not known per image). Scenario A2 injects extra pair flips into the
original labels (before the train/val split, like MNIST); y_reference then holds the original labels, so
the injected part of the noise is known and AUROC can be computed for it.
All images stay on the device (~0.7 GB as uint8); augmentation runs on the GPU.
"""
from pathlib import Path

import numpy as np
import torch

from src.noise import load_or_create_noise

CLASSES = ["cat", "lynx", "wolf", "coyote", "cheetah", "jaguar", "chimpanzee", "orangutan", "hamster",
           "guinea pig"]
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
RECORD = 4 + 4 + 3 * 64 * 64


def read_bin(path):
    """One .bin file -> (ids int64 [N], labels int64 [N], images uint8 [N, 3, 64, 64])."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % RECORD:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of the record size {RECORD}")
    raw = raw.reshape(-1, RECORD)
    for order in ("<", ">"):  # byte order is not documented; the labels must fall in 0..9
        ids = raw[:, :4].copy().view(f"{order}i4").ravel()
        labels = raw[:, 4:8].copy().view(f"{order}i4").ravel()
        if labels.min() >= 0 and labels.max() < len(CLASSES):
            break
    else:
        raise ValueError(f"{path}: labels outside 0..{len(CLASSES) - 1} in both byte orders")
    images = raw[:, 8:].reshape(-1, 3, 64, 64)
    return ids.astype(np.int64), labels.astype(np.int64), images


def read_image_folder(folder):
    """training/ or testing/ folder of "<label>_<name>.jpg" -> (names, labels int64 [N], images uint8 [N, 3, 64, 64])."""
    from PIL import Image
    files = sorted(Path(folder).glob("*.jpg"))
    labels = np.array([int(f.name.split("_")[0]) for f in files], dtype=np.int64)
    images = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).transpose(0, 3, 1, 2)
    return [f.name for f in files], labels, np.ascontiguousarray(images)


def load_tensors(root):
    """Cached (x_train, y_train, ids_train, x_test, y_test, ids_test)."""
    root = Path(root)
    cache = root / "animal10n.pt"
    if cache.exists():
        return torch.load(cache)
    train_dirs, test_dirs = sorted(root.rglob("training")), sorted(root.rglob("testing"))
    if train_dirs and test_dirs:
        out = {}
        for split, folder in (("train", train_dirs[0]), ("test", test_dirs[0])):
            names, labels, images = read_image_folder(folder)
            out.update({f"x_{split}": torch.from_numpy(images), f"y_{split}": torch.from_numpy(labels),
                        f"ids_{split}": names})
        torch.save(out, cache)
        return out
    train_files = sorted(root.rglob("data_batch_*.bin"))
    test_files = sorted(root.rglob("test_batch*.bin"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"ANIMAL-10N .bin files not found under {root} "
                                "(download via https://dm.kaist.ac.kr/datasets/animal-10n)")

    def read_all(files):
        parts = [read_bin(f) for f in files]
        return tuple(torch.from_numpy(np.concatenate([p[i] for p in parts])) for i in range(3))

    ids_tr, y_tr, x_tr = read_all(train_files)
    ids_te, y_te, x_te = read_all(test_files)
    out = {"x_train": x_tr, "y_train": y_tr, "ids_train": ids_tr,
           "x_test": x_te, "y_test": y_te, "ids_test": ids_te}
    torch.save(out, cache)
    return out


def make_transforms(mean, std, pad=4):
    """Train: per-image random crop (zero padding `pad`) + horizontal flip; both: normalization. On the GPU."""
    mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)

    def normalize(x):
        return (x.float() - mean.to(x.device)) / std.to(x.device)

    def train_tf(x):
        B, C, H, W = x.shape
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        rows = (i[:, None] + torch.arange(H, device=x.device))[:, None, :, None]     # [B, 1, H, 1]
        cols = (j[:, None] + torch.arange(W, device=x.device))[:, None, None, :]     # [B, 1, 1, W]
        b = torch.arange(B, device=x.device)[:, None, None, None]
        c = torch.arange(C, device=x.device)[None, :, None, None]
        x = xp[b, c, rows, cols]
        flip = torch.rand(B, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return normalize(x)

    return train_tf, normalize


def load_animal10n(cfg, device):
    """Returns (splits, noise_stats) in the format of src.data.load_dataset, plus transforms, class_names,
    pairs and ids. noise_stats is None without injected noise."""
    root = cfg.get("root", "data/animal10n")
    t = load_tensors(root)
    x, y = t["x_train"], t["y_train"]
    y_given, noise_stats = y, None
    if cfg.get("noise") is not None:
        y_given, noise_stats = load_or_create_noise(x, y, cfg["noise"], len(CLASSES), root, "animal10n", device)

    g = torch.Generator().manual_seed(cfg.get("split_seed", 0))
    perm = torch.randperm(x.shape[0], generator=g)
    n_val = int(round(cfg.get("val_fraction", 0.1) * x.shape[0]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def to_split(idx):
        return x[idx].to(device), y_given[idx].to(device), y[idx].to(device)

    splits = {"train": to_split(train_idx), "val": to_split(val_idx),
              "test": (t["x_test"].to(device), t["y_test"].to(device), t["y_test"].to(device))}
    xf = x[train_idx].float()
    mean, std = xf.mean(dim=(0, 2, 3)), xf.std(dim=(0, 2, 3))
    train_tf, eval_tf = make_transforms(mean, std, cfg.get("crop_padding", 4))
    splits.update(transforms={"train": train_tf, "eval": eval_tf}, class_names=CLASSES, pairs=PAIRS)
    return splits, noise_stats
