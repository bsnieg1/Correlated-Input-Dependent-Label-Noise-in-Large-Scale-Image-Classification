"""Output heads. Every head maps features h [B, D] to log p(y|x) [B, K].

Returning log-probabilities (not logits) keeps one loss for all methods: F.nll_loss.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Standard softmax classifier (CE baseline)."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, h):
        return F.log_softmax(self.fc(h).float(), dim=-1)


class HETHead(nn.Module):
    """Heteroscedastic output layer (Collier et al., CVPR 2021).

    Latent utilities u ~ N(mu(x), Sigma(x)) with low-rank + diagonal covariance
        Sigma(x) = V(x) V(x)^T + diag(d(x)^2),   V(x): [K, R],  d(x) > 0.
    Monte Carlo estimate with S samples via reparametrization:
        u_s = mu + V eps_R + d * eps_K,   eps ~ N(0, I)
        p(y|x) ~= (1/S) sum_s softmax(u_s / tau)
    R = 0 drops V and gives the diagonal variant (HET-diag).

    The log of the MC mean is computed as logsumexp(log_softmax) - log S, which is
    more stable than log(clip(mean(softmax))) used in edward2's MCSoftmaxDenseFA.
    """

    def __init__(self, in_dim, num_classes, num_factors=3, temperature=1.0, num_samples=1000,
                 min_scale=1e-3):
        super().__init__()
        self.K, self.R = num_classes, num_factors
        self.tau, self.S, self.min_scale = temperature, num_samples, min_scale
        self.mu = nn.Linear(in_dim, num_classes)
        self.d = nn.Linear(in_dim, num_classes)
        self.V = nn.Linear(in_dim, num_classes * num_factors) if num_factors > 0 else None

    def distribution_params(self, h):
        """Returns mu [B, K], V [B, K, R] (None if R = 0) and d [B, K]."""
        mu = self.mu(h)
        d = F.softplus(self.d(h)) + self.min_scale
        V = self.V(h).view(-1, self.K, self.R) if self.R > 0 else None
        return mu, V, d

    def covariance(self, h):
        """Sigma(x) = V V^T + diag(d^2), shape [B, K, K]. Used for interpreting class correlations."""
        mu, V, d = self.distribution_params(h.float())
        sigma = torch.diag_embed(d ** 2)
        if V is not None:
            sigma = sigma + V @ V.transpose(1, 2)
        return sigma

    def mean_log_probs(self, h):
        """'Clean' prediction log softmax(mu / tau), without noise (alternative prediction mode)."""
        return F.log_softmax(self.mu(h.float()) / self.tau, dim=-1)

    def sample_utilities(self, h, eps_K=None, eps_R=None):
        """u_s for each example, shape [B, S, K]. Noise can be passed in explicitly (tests)."""
        mu, V, d = self.distribution_params(h)
        B = h.shape[0]
        if eps_K is None:
            eps_K = torch.randn(B, self.S, self.K, device=h.device)
        u = mu[:, None] + d[:, None] * eps_K
        if V is not None:
            if eps_R is None:
                eps_R = torch.randn(B, self.S, self.R, device=h.device)
            u = u + torch.einsum("bkr,bsr->bsk", V, eps_R)
        return u

    def forward(self, h, eps_K=None, eps_R=None):
        # Kept in float32 even under AMP for numerical stability.
        with torch.autocast(device_type=h.device.type, enabled=False):
            u = self.sample_utilities(h.float(), eps_K, eps_R)
            log_p = F.log_softmax(u / self.tau, dim=-1)
            return torch.logsumexp(log_p, dim=1) - math.log(u.shape[1])


class HETDiag(HETHead):
    """HET with diagonal covariance only (R = 0): input-dependent variance, no class correlations."""

    def __init__(self, in_dim, num_classes, temperature=1.0, num_samples=1000, min_scale=1e-3):
        super().__init__(in_dim, num_classes, 0, temperature, num_samples, min_scale)


def build_head(cfg, in_dim, num_classes):
    name = cfg["name"]
    if name == "linear":
        return LinearHead(in_dim, num_classes)
    if name == "het":
        return HETHead(in_dim, num_classes, cfg.get("num_factors", 3), cfg.get("temperature", 1.0),
                       cfg.get("num_samples", 1000))
    if name == "het_diag":
        return HETDiag(in_dim, num_classes, cfg.get("temperature", 1.0), cfg.get("num_samples", 1000))
    raise ValueError(f"Unknown head: {name}")
