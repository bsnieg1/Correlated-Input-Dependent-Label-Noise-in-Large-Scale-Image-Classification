"""Report for the ANIMAL-10N experiments (results/animal_*/seed*/).

    python -m analysis.animal_report

A1 = original labels (~8% real human noise), A2 = A1 + 20% injected flips inside the 5 class pairs,
A3 = HET with R in {1, 3, 9} on A1 (R=5 is the main HET run, R=0 is HET-diag). The test set is the clean
ANIMAL-10N test set, used once after training. Writes results/tables/animal_*.{csv,md} and
results/figures/animal_*.png. All numbers are mean +- std over seeds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.aggregate import COLOR, INK, METHOD_LABEL, METHODS, MUTED, RESULTS, matrix_panel, ms, plt, \
    pooled_reliability, save, write_table
from src.animal_data import CLASSES, PAIRS

METHOD_LABEL = {**METHOD_LABEL, "het": "HET (R=5)"}  # ANIMAL configs use R=5 (MNIST main grid: R=3)
SCENARIOS = {"a1": "A1: original labels (~8% real noise)", "a2": "A2: + 20% pair flips"}
NAMES = ["cat", "lynx", "wolf", "coyote", "cheetah", "jaguar", "chimp", "orangutan", "hamster", "g. pig"]


def load():
    rows = []
    for p in sorted(RESULTS.glob("animal_*/seed*/summary.json")):
        s = json.loads(p.read_text())
        cfg = yaml.safe_load((p.parent / "config.yaml").read_text())
        exp = s["experiment"]
        scenario = "a2" if exp.startswith("animal_a2") else "a1"  # A3 runs use A1 labels
        R = cfg["head"].get("num_factors", 0) if cfg["head"]["name"] == "het" else None
        rows.append({"dir": p.parent, "experiment": exp, "scenario": scenario, "method": s["method"],
                     "ablation": exp.startswith("animal_a3"), "R": R, "seed": s["seed"], "s": s})
    return pd.DataFrame(rows)


def get(g, split, k):
    return pd.Series([s[split][k] for s in g.s])


def main_table(runs):
    rows = []
    for sc, sc_label in SCENARIOS.items():
        for m in METHODS + ["het_mu"]:
            base = "het" if m == "het_mu" else m
            g = runs[(runs.scenario == sc) & (runs.method == base) & ~runs.ablation]
            if g.empty:
                continue
            best, last = ("test_best_mu", "test_last_mu") if m == "het_mu" else ("test_best", "test_last")
            noise = pd.Series([(s.get("noise") or {}).get("train_noise_auroc", np.nan) for s in g.s])
            p5 = pd.Series([(s.get("correlation") or {}).get("pair_precision_at_k", np.nan) for s in g.s])
            rows.append({
                "scenario": sc_label, "method": METHOD_LABEL[m], "seeds": len(g),
                "top-1 best [%]": ms(get(g, best, "top1"), 100), "top-1 last [%]": ms(get(g, last, "top1"), 100),
                "drop best→last [p.p.]": ms(get(g, best, "top1") - get(g, last, "top1"), 100),
                "top-2 [%]": ms(get(g, best, "top2"), 100), "NLL": ms(get(g, best, "nll"), 1, 3),
                "ECE [%]": ms(get(g, best, "ece"), 100),
                "errors within pairs [%]": ms(get(g, best, "within_pair_error_share"), 100, 1),
                "AUROC (injected noise)": ms(noise, 1, 3) if m != "het_mu" else "-",
                "pair precision@5": ms(p5, 1) if m != "het_mu" else "-",
                "best epoch": f"{g.s.map(lambda s: s['best_epoch']).mean():.0f}",
            })
    write_table(pd.DataFrame(rows), "animal_main",
                "ANIMAL-10N, clean test set (best = epoch with the best noisy-val top-1)")


def significance_table(runs):
    """HET vs HET-diag and vs CE: difference of means against the 2-std rule of the thesis (§6)."""
    rows = []
    for sc, sc_label in SCENARIOS.items():
        for a, b in (("het", "het_diag"), ("het", "ce"), ("het_diag", "ce"), ("ls", "ce")):
            for split in ("test_best", "test_last"):
                ga = runs[(runs.scenario == sc) & (runs.method == a) & ~runs.ablation]
                gb = runs[(runs.scenario == sc) & (runs.method == b) & ~runs.ablation]
                if ga.empty or gb.empty:
                    continue
                xa, xb = get(ga, split, "top1") * 100, get(gb, split, "top1") * 100
                diff, sd = xa.mean() - xb.mean(), max(xa.std(ddof=1), xb.std(ddof=1))
                rows.append({"scenario": sc_label, "comparison": f"{METHOD_LABEL[a]} − {METHOD_LABEL[b]}",
                             "epoch": split.split("_")[1], "Δ top-1 [p.p.]": f"{diff:+.2f}",
                             "larger std [p.p.]": f"{sd:.2f}",
                             "verdict": "difference" if abs(diff) > 2 * sd else "within noise (< 2 std)"})
    write_table(pd.DataFrame(rows), "animal_significance", "ANIMAL-10N: is the difference larger than 2 std?")


def ablation_runs(runs):
    a1 = runs[runs.scenario == "a1"]
    abl = a1[a1.ablation | a1.method.isin(["het", "het_diag"])].copy()
    abl["R"] = abl.R.fillna(0).astype(int)
    return abl


def ablation_table(runs):
    abl = ablation_runs(runs)
    rows = []
    for R, g in abl.groupby("R"):
        corr = [s.get("correlation") or {} for s in g.s]
        rows.append({"R": R, "seeds": len(g), "top-1 best [%]": ms(get(g, "test_best", "top1"), 100),
                     "top-1 last [%]": ms(get(g, "test_last", "top1"), 100), "NLL": ms(get(g, "test_best", "nll"), 1, 3),
                     "ECE [%]": ms(get(g, "test_best", "ece"), 100),
                     "pair precision@5": ms(pd.Series([c.get("pair_precision_at_k", np.nan) for c in corr]), 1)
                     if R > 0 else "-",
                     "params": f"{g.s.iloc[0]['num_params']:,}".replace(",", " "),
                     "epoch time [s]": f"{g.s.map(lambda s: s['mean_epoch_time_s']).mean():.1f}"})
    write_table(pd.DataFrame(rows), "animal_ablation_r",
                "ANIMAL-10N A1: number of factors R (R=0 = HET-diag, R=5 = main HET; 5 class pairs)")


def cost_table(runs):
    rows = []
    for m in METHODS:
        g = runs[(runs.scenario == "a1") & (runs.method == m) & ~runs.ablation]
        rows.append({"method": METHOD_LABEL[m], "params": f"{g.s.iloc[0]['num_params']:,}".replace(",", " "),
                     "epoch time [s] (L4, bf16)": f"{g.s.map(lambda s: s['mean_epoch_time_s']).mean():.1f}"})
    write_table(pd.DataFrame(rows), "animal_cost", "ANIMAL-10N: cost (ResNet-18 for 64×64, batch 128, S=1000)")


# ---------------------------------------------------------------- figures

def fig_accuracy(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    w = 0.2
    for ax, (sc, label) in zip(axes, SCENARIOS.items()):
        for i, m in enumerate(METHODS):
            g = runs[(runs.scenario == sc) & (runs.method == m) & ~runs.ablation]
            for j, split in enumerate(("test_best", "test_last")):
                v = get(g, split, "top1") * 100
                x = j + (i - 1.5) * w
                ax.bar(x, v.mean(), w * 0.9, yerr=v.std(ddof=1), color=COLOR[m], capsize=2,
                       alpha=1 if j == 0 else 0.55, label=METHOD_LABEL[m] if j == 0 else None)
                ax.text(x, v.mean() + v.std(ddof=1) + 0.3, f"{v.mean():.1f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks([0, 1], ["best epoch (noisy val)", "last epoch (100)"])
        ax.set_title(label)
        ax.set_ylim(65, 88)
    axes[0].set_ylabel("clean test top-1 [%]")
    axes[0].legend(loc="lower left", ncol=2)
    fig.suptitle("ANIMAL-10N: clean test accuracy (mean ± std over 3 seeds)")
    fig.tight_layout()
    save(fig, "animal_accuracy")


def epoch_frame(g):
    frames = []
    for _, r in g.iterrows():
        e = pd.read_csv(Path(r["dir"]) / "epochs.csv")
        e["method"], e["seed"] = r["method"], r["seed"]
        frames.append(e)
    return pd.concat(frames)


def fig_curves(runs):
    for sc, label in SCENARIOS.items():
        ep = epoch_frame(runs[(runs.scenario == sc) & ~runs.ablation])
        panels = [("val_top1", "Val top-1 (noisy val labels)"),
                  ("train_acc_given", "Train top-1 w.r.t. training labels"),
                  ("train_loss", "Train loss (HET: MC estimate)")]
        if sc == "a2":
            panels = [panels[0], ("train_acc_clean", "Train top-1 w.r.t. original labels"),
                      ("memorized", "Flipped train images: predicts the flipped label"), panels[2]]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
        for ax, (col, title) in zip(axes, panels):
            for m in METHODS:
                g = ep[ep.method == m].groupby("epoch")[col]
                mean, std = g.mean(), g.std().fillna(0)
                ax.plot(mean.index, mean.values, color=COLOR[m], lw=1.6, label=METHOD_LABEL[m])
                ax.fill_between(mean.index, mean - std, mean + std, color=COLOR[m], alpha=0.15, lw=0)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("epoch")
        axes[0].legend()
        fig.suptitle(f"ANIMAL-10N {label}: learning curves (mean ± std over 3 seeds)")
        fig.tight_layout()
        save(fig, f"animal_curves_{sc}")


def mean_csv(g, name):
    return np.mean([np.loadtxt(Path(d) / name, delimiter=",") for d in g["dir"]], axis=0)


def fig_correlation(runs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4))
    for col, (sc, label) in enumerate(SCENARIOS.items()):
        het = runs[(runs.scenario == sc) & (runs.method == "het") & ~runs.ablation]
        corr = mean_csv(het, "correlation_test.csv")
        np.fill_diagonal(corr, np.nan)
        im = matrix_panel(axes[0, col], corr, "RdBu", -0.6, 0.6, f"{label}\nHET: mean class correlation from Σ (test)")
        ce = runs[(runs.scenario == sc) & (runs.method == "ce") & ~runs.ablation]
        conf = mean_csv(ce, "confusion_test.csv")
        conf = conf / conf.sum(axis=1, keepdims=True)
        np.fill_diagonal(conf, np.nan)
        im2 = matrix_panel(axes[1, col], conf, "Greys", 0, 0.2, "CE: test confusion (row-normalized, off-diagonal)",
                           min_show=0.01)
        for ax in axes[:, col]:
            ax.set_xticklabels(NAMES, rotation=60, fontsize=7)
            ax.set_yticklabels(NAMES, fontsize=7)
            for i, j in PAIRS:  # mark the 5 known pairs
                for a, b in ((i, j), (j, i)):
                    ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=INK, lw=1))
        fig.colorbar(im, ax=axes[0, col], shrink=0.8)
        fig.colorbar(im2, ax=axes[1, col], shrink=0.8)
    fig.suptitle("ANIMAL-10N: HET correlations vs CE confusions (framed cells = the 5 known class pairs)")
    fig.tight_layout()
    save(fig, "animal_correlation")


def fig_reliability(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (sc, label) in zip(axes, SCENARIOS.items()):
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
        for m in METHODS + ["het_mu"]:
            g = runs[(runs.scenario == sc) & (runs.method == ("het" if m == "het_mu" else m)) & ~runs.ablation]
            conf, acc = pooled_reliability(g, "mu" if m == "het_mu" else "mc")
            ax.plot(conf, acc, color=COLOR[m], marker="o", ms=3, lw=1.4 if m == "het_mu" else 2,
                    ls=":" if m == "het_mu" else "-", label=METHOD_LABEL[m])
        ax.set_title(label)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    fig.suptitle("ANIMAL-10N reliability (clean test, last epoch, bins pooled over seeds)")
    fig.tight_layout()
    save(fig, "animal_reliability")


def fig_ablation(runs):
    abl = ablation_runs(runs)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for split, ls, lab in (("test_best", "-", "best epoch"), ("test_last", "--", "last epoch")):
        stats = abl.groupby("R").apply(lambda g: pd.Series({"m": get(g, split, "top1").mean() * 100,
                                                             "s": get(g, split, "top1").std(ddof=1) * 100}))
        ax.errorbar(stats.index, stats.m, yerr=stats.s, color=COLOR["het"], ls=ls, marker="o", capsize=3, label=lab)
    ce = runs[(runs.scenario == "a1") & (runs.method == "ce") & ~runs.ablation]
    ax.axhline(get(ce, "test_best", "top1").mean() * 100, color=COLOR["ce"], lw=1, label="CE (best epoch)")
    ax.set_xticks([0, 1, 3, 5, 9])
    ax.set_xlabel("R (number of factors; 0 = HET-diag)")
    ax.set_ylabel("clean test top-1 [%]")
    ax.legend()
    ax.set_title("ANIMAL-10N A1: ablation of R")
    fig.tight_layout()
    save(fig, "animal_ablation_r")


def main():
    runs = load()
    print(runs.groupby("experiment").size().to_string())
    main_table(runs)
    significance_table(runs)
    ablation_table(runs)
    cost_table(runs)
    fig_accuracy(runs)
    fig_curves(runs)
    fig_correlation(runs)
    fig_reliability(runs)
    fig_ablation(runs)
    for name in ("animal_main", "animal_significance", "animal_ablation_r", "animal_cost"):
        print(f"wrote results/tables/{name}.md")


if __name__ == "__main__":
    main()
