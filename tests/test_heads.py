import math

import pytest
import torch
import torch.nn.functional as F

from src.heads import HETDiag, HETHead, build_head

D, K, R, B = 16, 10, 3, 8


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_output_shape_and_normalization():
    head = HETHead(D, K, R, num_samples=100)
    logp = head(torch.randn(B, D))
    assert logp.shape == (B, K)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(B), atol=1e-5)


def test_distribution_param_shapes():
    head = HETHead(D, K, R)
    mu, V, d = head.distribution_params(torch.randn(B, D))
    assert mu.shape == (B, K) and V.shape == (B, K, R) and d.shape == (B, K)
    assert (d > 0).all()


def test_num_params():
    # D*K*(R+2) weights + K*(R+2) biases (mu, d, V)
    head = HETHead(D, K, R)
    assert sum(p.numel() for p in head.parameters()) == (D + 1) * K * (R + 2)
    assert sum(p.numel() for p in HETDiag(D, K).parameters()) == (D + 1) * K * 2


def test_r0_equals_het_diag():
    full_r0, diag = HETHead(D, K, num_factors=0, num_samples=50), HETDiag(D, K, num_samples=50)
    diag.load_state_dict(full_r0.state_dict())
    h, eps_K = torch.randn(B, D), torch.randn(B, 50, K)
    assert full_r0.V is None and diag.V is None
    assert torch.allclose(full_r0(h, eps_K=eps_K), diag(h, eps_K=eps_K))


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_no_noise_reduces_to_softmax(tau):
    # V = 0 and d -> min_scale: every sample equals mu, so p(y|x) = softmax(mu / tau)
    head = HETHead(D, K, R, temperature=tau, num_samples=100, min_scale=0.0)
    with torch.no_grad():
        head.V.weight.zero_(); head.V.bias.zero_()
        head.d.weight.zero_(); head.d.bias.fill_(-30.0)  # softplus(-30) ~ 1e-13
    h = torch.randn(B, D)
    assert torch.allclose(head(h), F.log_softmax(head.mu(h) / tau, -1), atol=1e-5)
    assert torch.allclose(head.mean_log_probs(h), F.log_softmax(head.mu(h) / tau, -1), atol=1e-6)


def test_reparametrization_matches_manual_formula():
    S = 7
    head = HETHead(D, K, R, temperature=1.5, num_samples=S)
    h, eps_K, eps_R = torch.randn(B, D), torch.randn(B, S, K), torch.randn(B, S, R)
    mu, V, d = head.distribution_params(h)
    u = mu[:, None] + (V[:, None] @ eps_R[..., None]).squeeze(-1) + d[:, None] * eps_K
    expected = F.softmax(u / 1.5, -1).mean(1).log()
    assert torch.allclose(head(h, eps_K=eps_K, eps_R=eps_R), expected, atol=1e-5)


def test_sample_covariance_matches_sigma():
    head = HETHead(D, K, R, num_samples=200_000)
    h = torch.randn(1, D)
    with torch.no_grad():
        u = head.sample_utilities(h)[0]                        # [S, K]
        emp = torch.cov(u.T)
        sigma = head.covariance(h)[0]
    assert torch.allclose(emp, sigma, atol=0.05 * sigma.abs().max().item())


def test_covariance_is_symmetric_psd():
    sigma = HETHead(D, K, R).covariance(torch.randn(B, D))
    assert torch.allclose(sigma, sigma.transpose(1, 2))
    assert (torch.linalg.eigvalsh(sigma) > 0).all()


def test_gradients_reach_all_parameters():
    head = HETHead(D, K, R, num_samples=20)
    F.nll_loss(head(torch.randn(B, D)), torch.randint(0, K, (B,))).backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_stable_with_extreme_logits():
    head = HETHead(D, K, R, num_samples=50)
    logp = head(1e3 * torch.randn(B, D))
    assert torch.isfinite(logp).all()


def test_float32_output_from_half_input():
    head = HETHead(D, K, R, num_samples=10)
    assert head(torch.randn(B, D).half()).dtype == torch.float32


def test_build_head():
    assert isinstance(build_head({"name": "het", "num_factors": 5}, D, K), HETHead)
    assert build_head({"name": "het", "num_factors": 5}, D, K).R == 5
    assert isinstance(build_head({"name": "het_diag"}, D, K), HETDiag)


def test_mc_mean_is_close_to_large_s_estimate():
    # Small S estimate should agree with a large S estimate up to MC error.
    head = HETHead(D, K, R, num_samples=1000)
    h = torch.randn(2, D)
    small = head(h).exp()
    head.S = 100_000
    large = head(h).exp()
    assert torch.allclose(small, large, atol=0.03)
    assert math.isclose(large.sum().item(), 2.0, rel_tol=1e-5)
