"""Training entry point.

    python -m src.train --config configs/mnist_s0_ce.yaml --seed 0

Model selection uses the (noisy) validation set only. The clean test set is evaluated
once at the end, for the best-val checkpoint and for the last epoch.
Outputs go to results/<experiment>/seed<seed>/:
    config.yaml, epochs.csv, summary.json, training_curves.png,
    confusion_test.{csv,png}, reliability_test.{csv,png},
    HET heads (R > 0): correlation_test.{csv,png}, correlation_pairs_test.{csv,png},
    noisy runs: noise_transition.{csv,png}, noise_scores_train.png.
"""
import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data import TensorLoader, load_dataset
from src.heads import build_head
from src.metrics import (auroc, classification_metrics, confusion_matrix, covariance_to_correlation,
                         pair_correlation, pair_precision_at_k, reliability_bins)
from src.models import Classifier, build_backbone
from src.noise import MNIST_PAIRS
from src.plots import (plot_confusion, plot_correlation, plot_noise_scores, plot_reliability,
                       plot_training_curves, plot_transition)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(logp, y, label_smoothing=0.0):
    """NLL of the (noisy) label; with label smoothing a, the target is (1 - a) one-hot + a uniform."""
    loss = F.nll_loss(logp, y)
    if label_smoothing > 0:
        loss = (1 - label_smoothing) * loss - label_smoothing * logp.mean(dim=-1).mean()
    return loss


# Evaluation input pipeline, set at the start of every train() call: device, batch size, an optional
# transform (e.g. uint8 images kept on the CPU -> center crop + normalization on the GPU) and the AMP dtype.
EVAL = {"device": "cpu", "batch_size": 1000, "transform": None, "amp_dtype": None}


def autocast():
    """Mixed precision for the backbone when enabled (heads always compute in float32, see src.heads)."""
    dtype = EVAL["amp_dtype"]
    return torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None)


def to_memory_format(x):
    """channels_last for 4-D inputs when AMP is on (fast on tensor cores, ~6x slower on Pascal)."""
    return x.contiguous(memory_format=torch.channels_last) if EVAL["amp_dtype"] is not None and x.dim() == 4 else x


def amp_dtype(tcfg, device):
    """train.amp: true -> bfloat16 where supported (Ampere+), float16 with a gradient scaler on Volta/Turing,
    off on older GPUs (Pascal has no fast float16, AMP would be several times slower)."""
    if not tcfg.get("amp", False) or device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 7:
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def eval_inputs(x):
    """Batches of x ready for the model."""
    for i in range(0, x.shape[0], EVAL["batch_size"]):
        xb = x[i:i + EVAL["batch_size"]].to(EVAL["device"], non_blocking=True)
        xb = EVAL["transform"](xb) if EVAL["transform"] is not None else xb
        yield to_memory_format(xb)


@torch.no_grad()
def predict(model, x):
    model.eval()
    with autocast():
        return torch.cat([model(xb) for xb in eval_inputs(x)])


@torch.no_grad()
def predict_mu(model, x):
    """HET 'clean' prediction softmax(mu(x) / tau), without the noise term."""
    model.eval()
    with autocast():
        return torch.cat([model.head.mean_log_probs(model.backbone(xb)) for xb in eval_inputs(x)])


def train_label_stats(logp, y_noisy, y_clean):
    """Agreement of predictions with given vs true training labels; memorization on the noisy subset."""
    pred = logp.argmax(dim=-1)
    noisy = y_noisy != y_clean
    out = {"train_acc_given": (pred == y_noisy).float().mean().item(),
           "train_acc_clean": (pred == y_clean).float().mean().item(),
           "memorized": float("nan"), "corrected": float("nan")}
    if noisy.any():
        out["memorized"] = (pred[noisy] == y_noisy[noisy]).float().mean().item()   # fits the wrong label
        out["corrected"] = (pred[noisy] == y_clean[noisy]).float().mean().item()   # predicts the true class
    return out


def has_correlations(model):
    return getattr(model.head, "R", 0) > 0


@torch.no_grad()
def class_correlation_sums(model, x, y, num_classes):
    """Per-image class correlation from the HET covariance Sigma(x), summed per true class: [K, K, K]."""
    model.eval()
    total = torch.zeros(num_classes, num_classes, num_classes, device=y.device)
    bs = EVAL["batch_size"]
    for i, xb in zip(range(0, x.shape[0], bs), eval_inputs(x)):
        with autocast():
            h = model.backbone(xb)
        corr = covariance_to_correlation(model.head.covariance(h))
        total.index_add_(0, y[i:i + bs], corr)
    return total


def correlation_analysis(model, data, cfg, pairs, out_dir, best_epoch):
    """HET (R > 0) class correlations on the clean test set: global mean and pair-conditioned matrix.

    For clean data or symmetric noise the reference pairs serve as a negative control.
    """
    x, _, y = data["test"]
    K = cfg["data"]["num_classes"]
    sums = class_correlation_sums(model, x, y, K)
    corr_mean = (sums.sum(0) / x.shape[0]).cpu()
    corr_pair = pair_correlation(sums, torch.bincount(y, minlength=K).float()).cpu()
    np.savetxt(out_dir / "correlation_test.csv", corr_mean.numpy(), fmt="%.5f", delimiter=",")
    np.savetxt(out_dir / "correlation_pairs_test.csv", corr_pair.numpy(), fmt="%.5f", delimiter=",")
    names = data.get("class_names")
    plot_correlation(corr_mean.numpy(), out_dir / "correlation_test.png",
                     f"HET mean class correlation, clean test (epoch {best_epoch})", names)
    plot_correlation(corr_pair.numpy(), out_dir / "correlation_pairs_test.png",
                     f"HET correlation, avg over images of class i or j (epoch {best_epoch})", names)
    prec, found = pair_precision_at_k(corr_pair, pairs, len(pairs))
    off = corr_pair[~torch.eye(K, dtype=torch.bool)]
    return {"pair_precision_at_k": prec, "top_pairs": found,
            "pair_corr": {f"{i}-{j}": corr_pair[i, j].item() for i, j in pairs},
            "min_offdiag_corr": off.min().item()}


def noise_analysis(model, data, cfg, out_dir):
    """Runs with injected noise: AUROC of score = -log p(noisy label | x) for finding mislabeled training
    examples, plus figures of the score distributions and of the empirical noise transition matrix."""
    x, y_noisy, y_clean = data["train"]
    K = cfg["data"]["num_classes"]
    score = -predict(model, x).gather(1, y_noisy[:, None]).squeeze(1)
    is_noisy = y_noisy != y_clean
    T = torch.bincount(y_clean * K + y_noisy, minlength=K * K).view(K, K).float()
    T = (T / T.sum(1, keepdim=True)).cpu().numpy()
    np.savetxt(out_dir / "noise_transition.csv", T, fmt="%.4f", delimiter=",")
    plot_transition(T, out_dir / "noise_transition.png", "Label noise: P(given label | true class), train",
                    data.get("class_names"))
    score_auroc = auroc(score, is_noisy)
    plot_noise_scores(score.cpu().numpy(), is_noisy.cpu().numpy(), out_dir / "noise_scores_train.png",
                      f"-log p(given label | x) on train, AUROC = {score_auroc:.3f}")
    return {"train_noise_auroc": score_auroc}


def build_optimizer(cfg, model, steps_total):
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
                          weight_decay=cfg.get("weight_decay", 5e-4), nesterov=cfg.get("nesterov", False))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    return opt, sched


def train(cfg, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("output_dir", "results")) / cfg["experiment"] / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "seed": seed}, f, sort_keys=False)

    torch.backends.cudnn.benchmark = True
    data, noise_stats = load_dataset(cfg["data"], device)
    noise_cfg = cfg["data"].get("noise")
    pairs = data.get("pairs") or [tuple(p) for p in (noise_cfg or {}).get("pairs", MNIST_PAIRS)]
    transforms = data.get("transforms", {})
    train_tf = transforms.get("train")
    EVAL.update(device=device, batch_size=cfg["train"].get("eval_batch_size", 1000), transform=transforms.get("eval"),
                amp_dtype=amp_dtype(cfg["train"], device))
    scaler = torch.amp.GradScaler("cuda", enabled=EVAL["amp_dtype"] == torch.float16)
    if noise_stats is not None:
        print(f"label noise: train {(data['train'][1] != data['train'][2]).float().mean().item():.4f}, "
              f"val {(data['val'][1] != data['val'][2]).float().mean().item():.4f}")
    num_classes = cfg["data"]["num_classes"]
    backbone = build_backbone(cfg["model"])
    model = Classifier(backbone, build_head(cfg["head"], backbone.out_dim, num_classes)).to(device)
    if EVAL["amp_dtype"] is not None:
        model = model.to(memory_format=torch.channels_last)

    tcfg = cfg["train"]
    label_smoothing = tcfg.get("label_smoothing", 0.0)
    g = torch.Generator().manual_seed(seed)
    loader = TensorLoader(data["train"], tcfg["batch_size"], shuffle=True, generator=g)
    epochs = tcfg["epochs"]
    opt, sched = build_optimizer(tcfg, model, epochs * len(loader))

    best = {"val_top1": -math.inf, "epoch": None, "state": None}
    epoch_time_total = 0.0
    columns = ["epoch", "train_loss", "train_acc_given", "train_acc_clean", "memorized", "corrected",
               "val_top1", "val_nll", "lr", "epoch_time_s"]
    with open(out_dir / "epochs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            loss_sum, n = 0.0, 0
            for x, y, _ in loader:
                x, y = x.to(device, non_blocking=True), y.to(device)
                if train_tf is not None:
                    x = train_tf(x)
                x = to_memory_format(x)
                with autocast():
                    loss = loss_fn(model(x), y, label_smoothing)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                loss_sum += loss.detach() * x.shape[0]  # no per-step GPU sync
                n += x.shape[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - t0  # training only, without the evaluation below
            epoch_time_total += epoch_time

            x_tr, y_tr, y_tr_clean = data["train"]
            tr = train_label_stats(predict(model, x_tr), y_tr, y_tr_clean)
            val = classification_metrics(predict(model, data["val"][0]), data["val"][1])
            row = {"epoch": epoch, "train_loss": float(loss_sum) / n, **tr, "val_top1": val["top1"],
                   "val_nll": val["nll"], "lr": sched.get_last_lr()[0], "epoch_time_s": epoch_time}
            writer.writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in columns])
            f.flush()
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  train acc given/clean "
                  f"{tr['train_acc_given']:.4f}/{tr['train_acc_clean']:.4f}  memorized {tr['memorized']:.3f}  "
                  f"val top1 {val['top1']:.4f}  ({epoch_time:.1f}s)")
            if val["top1"] > best["val_top1"]:
                best = {"val_top1": val["top1"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    # Clean test set: evaluated only here, once per checkpoint.
    x_test, y_test, _ = data["test"]
    is_het = hasattr(model.head, "mean_log_probs")
    summary = {"experiment": cfg["experiment"], "seed": seed, "method": cfg.get("method", cfg["head"]["name"])}
    summary["test_last"] = classification_metrics(predict(model, x_test), y_test, pairs)
    if is_het:
        summary["test_last_mu"] = classification_metrics(predict_mu(model, x_test), y_test, pairs)
    model.load_state_dict(best["state"])
    logp_test = predict(model, x_test)
    summary["test_best"] = classification_metrics(logp_test, y_test, pairs)
    if is_het:
        logp_mu = predict_mu(model, x_test)
        summary["test_best_mu"] = classification_metrics(logp_mu, y_test, pairs)

    # Figures and matrices for the best-val checkpoint.
    best_epoch = best["epoch"]
    cm = confusion_matrix(logp_test, y_test, num_classes).cpu().numpy()
    np.savetxt(out_dir / "confusion_test.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, out_dir / "confusion_test.png", f"Confusion matrix, clean test (epoch {best_epoch})",
                   data.get("class_names"))
    plot_training_curves(out_dir / "epochs.csv", out_dir / "training_curves.png", best_epoch)
    reliability = {"mc": reliability_bins(logp_test, y_test)}
    if is_het:
        reliability["mu"] = reliability_bins(logp_mu, y_test)
    with open(out_dir / "reliability_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "bin", "confidence", "accuracy", "count"])
        for mode, rows in reliability.items():
            writer.writerows([mode, b, *r] for b, r in enumerate(rows))
    plot_reliability(reliability, out_dir / "reliability_test.png",
                     f"Reliability, clean test, ECE = {summary['test_best']['ece']:.3f}")

    summary.update({
        "best_epoch": best_epoch, "best_val_top1": best["val_top1"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "mean_epoch_time_s": epoch_time_total / epochs,
    })
    if has_correlations(model):
        summary["correlation"] = correlation_analysis(model, data, cfg, pairs, out_dir, best_epoch)
    if noise_stats is not None:
        summary["noise"] = {**noise_stats,
                            "train_rate": (data["train"][1] != data["train"][2]).float().mean().item(),
                            "val_rate": (data["val"][1] != data["val"][2]).float().mean().item(),
                            **noise_analysis(model, data, cfg, out_dir)}
    if "ids" in data:  # per-image outputs of the best checkpoint, for error analysis
        arrays = {"test_ids": np.array(data["ids"]["test"]), "test_logp": logp_test.cpu().numpy(),
                  "train_ids": np.array(data["ids"]["train"]), "train_logp": predict(model, x_tr).cpu().numpy()}
        if is_het:
            arrays["test_logp_mu"] = logp_mu.cpu().numpy()
        np.savez_compressed(out_dir / "predictions.npz", **arrays)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("test_best", "test_last", "best_epoch")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.seed)


if __name__ == "__main__":
    main()
