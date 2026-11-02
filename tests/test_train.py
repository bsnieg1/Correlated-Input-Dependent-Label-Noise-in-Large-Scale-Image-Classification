import torch
import torch.nn.functional as F

from src.train import loss_fn, train_label_stats


def test_label_smoothing_matches_pytorch():
    logits = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    ours = loss_fn(F.log_softmax(logits, -1), y, label_smoothing=0.1)
    ref = F.cross_entropy(logits, y, label_smoothing=0.1)
    assert torch.allclose(ours, ref, atol=1e-6)
    assert torch.allclose(loss_fn(F.log_softmax(logits, -1), y), F.cross_entropy(logits, y))


def test_train_label_stats():
    y_clean = torch.tensor([0, 1, 2, 3])
    y_noisy = torch.tensor([0, 1, 5, 6])      # last two mislabeled
    pred = torch.tensor([0, 1, 5, 3])         # memorizes one, corrects one
    s = train_label_stats(F.one_hot(pred, 10).float().log(), y_noisy, y_clean)
    assert s["train_acc_given"] == 0.75 and s["train_acc_clean"] == 0.75
    assert s["memorized"] == 0.5 and s["corrected"] == 0.5
