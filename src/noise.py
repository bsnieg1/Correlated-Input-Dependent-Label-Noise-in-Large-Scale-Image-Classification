"""Synthetic label noise (MNIST scenarios S1-S3). Only labels change; images stay intact.

    pair       S1: class-dependent flips inside fixed pairs (1<->7, 4<->9, 3<->8)
    instance   S2: same pairs, but P(flip i -> j | x) is proportional to p_aux(j | x)
               from a clean auxiliary CE model (harder images are flipped more often)
    symmetric  S3: negative control, flips to a uniformly random other class

Noise is applied once to the full training set (before the train/val split), with its own
seed, and cached in <root>/noise/ so every method and training seed sees identical labels.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

MNIST_PAIRS = [(1, 7), (4, 9), (3, 8)]
# Noise RNG streams are offset from the noise seed: with equal seeds, torch.randperm(N) here would
# reproduce the train/val split permutation exactly (e.g. S3 would flip the whole validation set).
SEED_OFFSET = 1_000_003
CACHE_VERSION = 2  # bump when the noise generation changes, to invalidate cached labels


def _directed(pairs):
    return [(i, j) for a, b in pairs for i, j in ((a, b), (b, a))]


def pair_noise(y, pairs, rate, g):
    """S1: inside each pair (i, j), exactly round(rate * n_i) examples of class i get label j, and vice versa."""
    y_noisy = y.clone()
    for src, dst in _directed(pairs):
        idx = (y == src).nonzero().squeeze(1)
        chosen = idx[torch.randperm(idx.numel(), generator=g)[:round(rate * idx.numel())]]
        y_noisy[chosen] = dst
    return y_noisy


def symmetric_noise(y, rate, num_classes, g):
    """S3: exactly round(rate * N) examples get a uniformly random class different from the true one."""
    n_flip = round(rate * y.numel())
    chosen = torch.randperm(y.numel(), generator=g)[:n_flip]
    offset = torch.randint(1, num_classes, (n_flip,), generator=g)
    y_noisy = y.clone()
    y_noisy[chosen] = (y[chosen] + offset) % num_classes
    return y_noisy


def scaled_flip_probs(weights, rate, max_prob):
    """p = min(max_prob, c * w) with c found by bisection so that mean(p) = rate."""
    weights = weights.double()
    if rate >= max_prob * (weights > 0).double().mean():
        raise ValueError(f"Cannot reach mean flip rate {rate} with max_prob {max_prob}")
    probs = lambda c: (c * weights).clamp(max=max_prob)
    lo, hi = 0.0, 1.0
    while probs(hi).mean() < rate:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if probs(mid).mean() < rate else (lo, mid)
    return probs(hi).float()


def instance_pair_noise(y, aux_probs, pairs, rate, max_prob, g):
    """S2: for an image of class i, P(label j) = min(max_prob, c * p_aux(j | x)), averaging `rate` per class.

    Returns noisy labels and the per-example flip probabilities.
    """
    y_noisy, flip_prob = y.clone(), torch.zeros(y.shape[0])
    for src, dst in _directed(pairs):
        idx = (y == src).nonzero().squeeze(1)
        p = scaled_flip_probs(aux_probs[idx, dst], rate, max_prob)
        flip_prob[idx] = p
        y_noisy[idx[torch.rand(p.shape, generator=g) < p]] = dst
    return y_noisy, flip_prob


def auxiliary_probs(x, y, num_classes, cfg, seed, device):
    """Out-of-fold p_aux(. | x) [N, K] from clean CE models (cross-fitting: no image is scored by a model that saw it).

    A softmax temperature > 1 spreads the (typically over-confident) probabilities so flips are not
    concentrated on a handful of images.
    """
    from src.heads import LinearHead
    from src.models import Classifier, SmallCNN

    folds = cfg.get("aux_folds", 2)
    epochs, batch_size, lr = cfg.get("aux_epochs", 3), cfg.get("aux_batch_size", 128), cfg.get("aux_lr", 0.05)
    g = torch.Generator().manual_seed(SEED_OFFSET + seed + 1)
    fold_of = torch.randperm(x.shape[0], generator=g) % folds
    logits = torch.empty(x.shape[0], num_classes)
    for k in range(folds):
        torch.manual_seed(seed + k)
        fit = (fold_of != k).nonzero().squeeze(1)
        model = Classifier(SmallCNN(x.shape[1]), LinearHead(128, num_classes)).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        steps = epochs * ((fit.numel() + batch_size - 1) // batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        model.train()
        for _ in range(epochs):
            perm = fit[torch.randperm(fit.numel(), generator=g)]
            for start in range(0, perm.numel(), batch_size):
                b = perm[start:start + batch_size]
                loss = F.nll_loss(model(x[b].to(device)), y[b].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        held = (fold_of == k).nonzero().squeeze(1)
        with torch.no_grad():
            for start in range(0, held.numel(), 1000):
                b = held[start:start + 1000]
                logits[b] = model(x[b].to(device)).cpu()  # log-probs = logits up to a per-row shift
    return F.softmax(logits / cfg.get("aux_temperature", 1.0), dim=-1)


def apply_noise(x, y, cfg, num_classes, device):
    """Returns noisy labels and a dict of noise statistics."""
    kind, rate, seed = cfg["type"], cfg["rate"], cfg.get("seed", 0)
    pairs = [tuple(p) for p in cfg.get("pairs", MNIST_PAIRS)]
    g = torch.Generator().manual_seed(SEED_OFFSET + seed)
    stats = {}
    if kind == "pair":
        y_noisy = pair_noise(y, pairs, rate, g)
    elif kind == "symmetric":
        y_noisy = symmetric_noise(y, rate, num_classes, g)
    elif kind == "instance":
        aux = auxiliary_probs(x, y, num_classes, cfg, seed, device)
        y_noisy, flip_prob = instance_pair_noise(y, aux, pairs, rate, cfg.get("max_prob", 0.5), g)
        p = flip_prob[torch.isin(y, torch.tensor([c for pair in pairs for c in pair]))]
        stats.update(flip_prob_mean=p.mean().item(), flip_prob_median=p.median().item(),
                     flip_prob_at_max=(p >= cfg.get("max_prob", 0.5) - 1e-6).float().mean().item())
    else:
        raise ValueError(f"Unknown noise type: {kind}")
    stats["noise_rate"] = (y_noisy != y).float().mean().item()
    stats["transitions"] = {f"{i}->{j}": ((y == i) & (y_noisy == j)).sum().item() / max((y == i).sum().item(), 1)
                            for i in range(num_classes) for j in range(num_classes)
                            if i != j and ((y == i) & (y_noisy == j)).any()}
    return y_noisy, stats


def load_or_create_noise(x, y, cfg, num_classes, root, dataset, device):
    """Cached apply_noise: the cache key is the full noise config."""
    key = hashlib.sha1(json.dumps({**cfg, "_version": CACHE_VERSION}, sort_keys=True).encode()).hexdigest()[:10]
    path = Path(root) / "noise" / f"{dataset}_{cfg['type']}_{key}.pt"
    if path.exists():
        cached = torch.load(path)
        return cached["y_noisy"], cached["stats"]
    y_noisy, stats = apply_noise(x, y, cfg, num_classes, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"y_noisy": y_noisy, "stats": stats, "config": cfg}, path)
    return y_noisy, stats
