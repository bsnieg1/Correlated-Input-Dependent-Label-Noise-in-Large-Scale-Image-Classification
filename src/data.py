"""Datasets kept entirely on the device as tensors (MNIST fits easily in GPU memory).

Each split is a tuple (x, y, y_clean): y are the (possibly noisy) training labels,
y_clean the ground truth. Label noise (src/noise.py) modifies y of train and val only;
the test set stays clean.
"""
import torch
from torchvision import datasets

from src.noise import load_or_create_noise

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class TensorLoader:
    """Minimal batch iterator over tensors already on the target device."""

    def __init__(self, tensors, batch_size, shuffle, generator=None):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.n = tensors[0].shape[0]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        device = self.tensors[0].device
        if self.shuffle:
            idx = torch.randperm(self.n, generator=self.generator).to(device)
        else:
            idx = torch.arange(self.n, device=device)
        for start in range(0, self.n, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield tuple(t[b] for t in self.tensors)


def _mnist_tensors(root, train):
    ds = datasets.MNIST(root, train=train, download=True)
    x = ds.data.float().div_(255).sub_(MNIST_MEAN).div_(MNIST_STD).unsqueeze(1)  # [N, 1, 28, 28]
    return x, ds.targets.clone()


def load_mnist(root, val_fraction, split_seed, device, noise=None, num_classes=10):
    """Returns (splits, noise_stats); splits: dict with 'train', 'val', 'test' -> (x, y, y_clean).

    Noise is applied to all 60k training images before the split, so train and val are equally noisy.
    The train/val split is fixed by split_seed (independent of the training seed),
    so every method and seed sees the same validation set.
    """
    x, y = _mnist_tensors(root, train=True)
    y_noisy, noise_stats = y, None
    if noise is not None:
        y_noisy, noise_stats = load_or_create_noise(x, y, noise, num_classes, root, "mnist", device)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_val = int(round(val_fraction * x.shape[0]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_test, y_test = _mnist_tensors(root, train=False)

    def to_split(idx):
        return x[idx].to(device), y_noisy[idx].to(device), y[idx].to(device)

    splits = {
        "train": to_split(train_idx),
        "val": to_split(val_idx),
        "test": (x_test.to(device), y_test.to(device), y_test.clone().to(device)),
    }
    return splits, noise_stats


def load_dataset(cfg, device):
    name = cfg["name"]
    if name == "mnist":
        return load_mnist(cfg.get("root", "data"), cfg.get("val_fraction", 0.1), cfg.get("split_seed", 0),
                          device, cfg.get("noise"), cfg["num_classes"])
    raise ValueError(f"Unknown dataset: {name}")
