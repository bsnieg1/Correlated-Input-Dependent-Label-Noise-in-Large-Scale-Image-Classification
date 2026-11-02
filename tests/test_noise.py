import math

import pytest
import torch

from src.noise import MNIST_PAIRS, instance_pair_noise, pair_noise, scaled_flip_probs, symmetric_noise


@pytest.fixture
def labels():
    return torch.arange(10).repeat_interleave(1000)  # 1000 examples per class


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.45])
def test_pair_noise_exact_rate_and_only_within_pairs(labels, rate):
    y_noisy = pair_noise(labels, MNIST_PAIRS, rate, torch.Generator().manual_seed(0))
    for i, j in MNIST_PAIRS:
        for src, dst in ((i, j), (j, i)):
            assert (y_noisy[labels == src] == dst).sum().item() == round(rate * 1000)
    flipped = y_noisy != labels
    allowed = {(i, j) for a, b in MNIST_PAIRS for i, j in ((a, b), (b, a))}
    assert all((a, b) in allowed for a, b in zip(labels[flipped].tolist(), y_noisy[flipped].tolist()))
    for c in (0, 2, 5, 6):  # classes outside pairs are untouched
        assert torch.equal(y_noisy[labels == c], labels[labels == c])


def test_pair_noise_reproducible(labels):
    a = pair_noise(labels, MNIST_PAIRS, 0.3, torch.Generator().manual_seed(1))
    b = pair_noise(labels, MNIST_PAIRS, 0.3, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_symmetric_noise(labels):
    y_noisy = symmetric_noise(labels, 0.3, 10, torch.Generator().manual_seed(0))
    flipped = y_noisy != labels
    assert flipped.sum().item() == 3000  # always to a different class
    # Targets are roughly uniform over the 9 other classes: no preferred pairs.
    T = torch.zeros(10, 10)
    T.index_put_((labels[flipped], y_noisy[flipped]), torch.ones(3000), accumulate=True)
    off = T[~torch.eye(10, dtype=torch.bool)]
    assert off.min() > 15 and off.max() < 55  # mean 33.3 per cell


def test_scaled_flip_probs_mean_and_cap():
    w = torch.rand(5000) ** 4  # heavy-tailed weights
    p = scaled_flip_probs(w, rate=0.3, max_prob=0.5)
    assert math.isclose(p.mean().item(), 0.3, abs_tol=1e-4)
    assert p.max().item() <= 0.5 + 1e-6
    # Monotone in the weight: images that look more like the partner class flip more often.
    order = w.argsort()
    assert (p[order].diff() >= -1e-7).all()


def test_scaled_flip_probs_impossible_rate():
    with pytest.raises(ValueError):
        scaled_flip_probs(torch.tensor([1.0, 0.0, 0.0, 0.0]), rate=0.3, max_prob=0.5)


def test_instance_pair_noise_follows_aux_probs(labels):
    g = torch.Generator().manual_seed(0)
    # Fake auxiliary model: for each image, a random probability of looking like its partner class.
    aux = torch.full((labels.numel(), 10), 1e-3)
    looks_like_partner = torch.rand(labels.numel(), generator=g)
    for src, dst in [(1, 7), (7, 1), (4, 9), (9, 4), (3, 8), (8, 3)]:
        aux[labels == src, dst] = looks_like_partner[labels == src]
    y_noisy, flip_prob = instance_pair_noise(labels, aux, MNIST_PAIRS, 0.3, 0.5, g)
    in_pairs = torch.isin(labels, torch.tensor([1, 7, 4, 9, 3, 8]))
    assert math.isclose(flip_prob[in_pairs].mean().item(), 0.3, abs_tol=1e-3)
    assert (flip_prob[~in_pairs] == 0).all() and torch.equal(y_noisy[~in_pairs], labels[~in_pairs])
    rate = (y_noisy != labels)[in_pairs].float().mean().item()
    assert abs(rate - 0.3) < 0.02  # Bernoulli draws around the target rate
    # Flipped images have, on average, higher partner probability than the kept ones.
    flipped = (y_noisy != labels) & in_pairs
    kept = (y_noisy == labels) & in_pairs
    assert looks_like_partner[flipped].mean() > looks_like_partner[kept].mean() + 0.1


def test_noise_independent_of_train_val_split():
    """Regression: symmetric noise with seed 0 used to flip exactly the split-seed-0 validation set."""
    from src.noise import apply_noise
    y = torch.arange(10).repeat(6000)  # 60k labels
    y_noisy, _ = apply_noise(None, y, {"type": "symmetric", "rate": 0.3, "seed": 0}, 10, "cpu")
    val_idx = torch.randperm(60000, generator=torch.Generator().manual_seed(0))[:6000]
    val_rate = (y_noisy[val_idx] != y[val_idx]).float().mean().item()
    assert abs(val_rate - 0.3) < 0.03
