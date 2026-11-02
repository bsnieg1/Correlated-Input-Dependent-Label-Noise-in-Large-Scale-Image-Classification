"""Evaluation metrics computed from log-probabilities logp [N, K] and labels y [N]."""
import torch
import torch.nn.functional as F


def top_k_accuracy(logp, y, k=1):
    topk = logp.topk(k, dim=-1).indices
    return (topk == y[:, None]).any(dim=-1).float().mean().item()


def nll(logp, y):
    return F.nll_loss(logp, y).item()


def reliability_bins(logp, y, n_bins=15):
    """Per equal-width confidence bin: (mean confidence, accuracy, count). Empty bins give NaN."""
    conf, pred = logp.exp().max(dim=-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=logp.device)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        n = in_bin.sum().item()
        rows.append((conf[in_bin].mean().item() if n else float("nan"),
                     correct[in_bin].mean().item() if n else float("nan"), n))
    return rows


def ece(logp, y, n_bins=15):
    """Expected Calibration Error (Guo et al. 2017), equal-width confidence bins."""
    n_total = y.shape[0]
    return sum(n / n_total * abs(c - a) for c, a, n in reliability_bins(logp, y, n_bins) if n)


def pair_metrics(logp, y, pairs):
    """Top-1 on test images from pair classes, and the share of all errors that stay inside a pair."""
    pred = logp.argmax(dim=-1)
    partner = torch.full((logp.shape[1],), -1, device=y.device)
    for i, j in pairs:
        partner[i], partner[j] = j, i
    in_pairs = partner[y] >= 0
    wrong = pred != y
    return {
        "pair_top1": (~wrong[in_pairs]).float().mean().item(),
        "other_top1": (~wrong[~in_pairs]).float().mean().item(),
        "within_pair_error_share": ((pred == partner[y]) & wrong).sum().item() / max(wrong.sum().item(), 1),
    }


def confusion_matrix(logp, y, num_classes):
    """C[i, j] = number of examples with true class i predicted as j."""
    pred = logp.argmax(dim=-1)
    return torch.bincount(y * num_classes + pred, minlength=num_classes ** 2).view(num_classes, num_classes)


def covariance_to_correlation(sigma):
    """corr_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj), batched over leading dims."""
    std = sigma.diagonal(dim1=-2, dim2=-1).sqrt()
    return sigma / (std[..., :, None] * std[..., None, :])


def classification_metrics(logp, y, pairs=None):
    out = {
        "top1": top_k_accuracy(logp, y, 1),
        "top2": top_k_accuracy(logp, y, 2),
        "nll": nll(logp, y),
        "ece": ece(logp, y),
    }
    if pairs:
        out.update(pair_metrics(logp, y, pairs))
    return out


def auroc(scores, positive):
    """Area under ROC = P(score of a random positive > score of a random negative), ties count 1/2."""
    scores, positive = scores.double().cpu(), positive.bool().cpu()
    pos, neg = scores[positive], scores[~positive]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    ranks = torch.empty_like(scores)
    order = scores.argsort()
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.double)
    # Average ranks within ties.
    uniq, inverse = scores.unique(return_inverse=True)
    ranks = torch.zeros(uniq.numel(), dtype=torch.double).index_add_(0, inverse, ranks)[inverse] \
        / torch.bincount(inverse).double()[inverse]
    n_pos, n_neg = pos.numel(), neg.numel()
    return ((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def pair_correlation(corr_sum_by_class, counts):
    """C_ij = mean of corr_ij(x) over images whose true class is i or j.

    corr_sum_by_class [K, K, K]: sum of per-image correlation matrices, grouped by true class.
    counts [K]: number of images per true class.
    """
    S = corr_sum_by_class
    from_i = S.diagonal(dim1=0, dim2=1).T        # [i, j] -> S[i, i, j]
    from_j = S.diagonal(dim1=0, dim2=2)          # [i, j] -> S[j, i, j]
    return (from_i + from_j) / (counts[:, None] + counts[None, :])


def pair_precision_at_k(corr, true_pairs, k):
    """Fraction of the k most negatively correlated class pairs (i < j) that are true noise pairs."""
    K = corr.shape[0]
    iu = torch.triu_indices(K, K, offset=1)
    vals = corr[iu[0], iu[1]]
    top = vals.argsort()[:k]
    truth = {tuple(sorted(p)) for p in true_pairs}
    found = [(iu[0, t].item(), iu[1, t].item()) for t in top]
    return sum(p in truth for p in found) / k, found
