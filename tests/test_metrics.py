import math

import torch

from src.heads import LinearHead
from src.metrics import (auroc, confusion_matrix, covariance_to_correlation, ece, nll, pair_correlation,
                         pair_precision_at_k, top_k_accuracy)


def test_confusion_matrix_hand_example():
    y = torch.tensor([0, 0, 1, 2, 2])
    pred = torch.tensor([0, 1, 1, 2, 0])
    logp = torch.nn.functional.one_hot(pred, 3).float().add(1e-3).log()
    expected = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    assert torch.equal(confusion_matrix(logp, y, 3), expected)


def test_covariance_to_correlation():
    sigma = torch.tensor([[4.0, -2.0], [-2.0, 9.0]])
    corr = covariance_to_correlation(sigma)
    assert torch.allclose(corr, torch.tensor([[1.0, -1 / 3], [-1 / 3, 1.0]]))


def test_top_k_accuracy_hand_example():
    p = torch.tensor([[0.7, 0.2, 0.1],   # pred 0, second 1
                      [0.1, 0.3, 0.6],   # pred 2, second 1
                      [0.5, 0.1, 0.4],   # pred 0, second 2
                      [0.2, 0.5, 0.3]])  # pred 1, second 2
    y = torch.tensor([0, 1, 2, 0])
    logp = p.log()
    assert top_k_accuracy(logp, y, 1) == 0.25
    assert top_k_accuracy(logp, y, 2) == 0.75


def test_nll_uniform():
    logp = torch.full((5, 4), -math.log(4))
    assert abs(nll(logp, torch.zeros(5, dtype=torch.long)) - math.log(4)) < 1e-6


def test_ece_hand_example():
    # All predictions have confidence 0.8, half are correct -> ECE = |0.8 - 0.5| = 0.3
    p = torch.tensor([[0.8, 0.2]] * 4)
    y = torch.tensor([0, 0, 1, 1])
    assert abs(ece(p.log(), y) - 0.3) < 1e-6


def test_ece_perfect_calibration():
    p = torch.tensor([[1.0, 0.0]] * 3).clamp_min(1e-12)
    y = torch.zeros(3, dtype=torch.long)
    assert ece(p.log(), y) < 1e-6


def test_linear_head_outputs_log_probs():
    head = LinearHead(16, 10)
    logp = head(torch.randn(8, 16))
    assert logp.shape == (8, 10)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(8), atol=1e-5)


def test_auroc_hand_example():
    # 3 positives, 3 negatives; 8 of 9 (pos, neg) pairs are ordered correctly -> 8/9.
    scores = torch.tensor([0.9, 0.8, 0.3, 0.4, 0.2, 0.1])
    positive = torch.tensor([1, 1, 1, 0, 0, 0])
    assert math.isclose(auroc(scores, positive), 8 / 9)


def test_auroc_ties_and_extremes():
    assert auroc(torch.tensor([1.0, 1.0]), torch.tensor([1, 0])) == 0.5
    assert auroc(torch.tensor([2.0, 1.0]), torch.tensor([1, 0])) == 1.0
    assert auroc(torch.tensor([1.0, 2.0]), torch.tensor([1, 0])) == 0.0


def test_pair_correlation_averages_over_both_classes():
    K = 3
    sums = torch.zeros(K, K, K)
    sums[0, 0, 1] = sums[0, 1, 0] = -2.0     # 2 images of class 0, corr_01 = -1 each
    sums[1, 0, 1] = sums[1, 1, 0] = 0.0      # 2 images of class 1, corr_01 = 0 each
    counts = torch.tensor([2.0, 2.0, 1.0])
    C = pair_correlation(sums, counts)
    assert math.isclose(C[0, 1].item(), -0.5)
    assert torch.allclose(C, C.T)


def test_pair_precision_at_k():
    corr = torch.eye(4)
    corr[0, 1] = corr[1, 0] = -0.9
    corr[2, 3] = corr[3, 2] = -0.5
    corr[0, 2] = corr[2, 0] = -0.7
    prec, found = pair_precision_at_k(corr, [(1, 0), (2, 3)], k=2)
    assert prec == 0.5 and found == [(0, 1), (0, 2)]
    assert pair_precision_at_k(corr, [(0, 1), (2, 3)], k=3)[0] == 2 / 3


def test_pair_metrics_hand_example():
    from src.metrics import pair_metrics
    y = torch.tensor([1, 7, 7, 0, 0, 3])
    pred = torch.tensor([1, 1, 7, 0, 5, 8])   # errors: 7->1 (pair), 0->5 (not), 3->8 (pair)
    logp = torch.nn.functional.one_hot(pred, 10).float().add(1e-3).log()
    m = pair_metrics(logp, y, [(1, 7), (3, 8)])
    assert math.isclose(m["pair_top1"], 2 / 4)
    assert math.isclose(m["other_top1"], 1 / 2)
    assert math.isclose(m["within_pair_error_share"], 2 / 3)
