"""Aggregates MNIST runs (results/mnist_*/seed*/) into tables and comparison figures.

    python -m analysis.aggregate

Writes results/tables/*.{csv,md} and results/figures/*.png. All numbers are mean +- std over seeds.
Test-set numbers are reported only here, after all hyperparameters were fixed on the noisy val set.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

RESULTS = Path("results")
TABLES, FIGURES = RESULTS / "tables", RESULTS / "figures"

SCENARIOS = ["s0", "s1_p10", "s1_p30", "s1_p45", "s2_p30", "s3_p30"]
SCENARIO_LABEL = {"s0": "S0 clean", "s1_p10": "S1 pairs 10%", "s1_p30": "S1 pairs 30%",
                  "s1_p45": "S1 pairs 45%", "s2_p30": "S2 image-dep. 30%", "s3_p30": "S3 symmetric 30%"}
NOISY = SCENARIOS[1:]
METHODS = ["ce", "ls", "het_diag", "het"]
METHOD_LABEL = {"ce": "CE", "ls": "LS (0.1)", "het_diag": "HET-diag (R=0)", "het": "HET (R=3)",
                "het_mu": "HET, softmax(μ)"}
# Fixed categorical slots of the validated reference palette (dataviz skill), assigned once per method.
COLOR = {"het": "#2a78d6", "het_diag": "#eb6834", "ce": "#1baf7a", "ls": "#eda100", "het_mu": "#2a78d6"}
PAIRS = [(1, 7), (4, 9), (3, 8)]
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "font.size": 9, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- loading

def load_runs():
    rows = []
    for summary_path in sorted(RESULTS.glob("mnist_*/seed*/summary.json")):
        run_dir = summary_path.parent
        s = json.loads(summary_path.read_text())
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        exp = s["experiment"]
        method = cfg.get("method", cfg["head"]["name"])
        ablation = exp.startswith("mnist_abl_")
        scenario = "s1_p30" if ablation else exp[len("mnist_"):-len(method) - 1]
        row = {"experiment": exp, "scenario": scenario, "method": method, "seed": s["seed"],
               "ablation": ablation, "R": cfg["head"].get("num_factors", 0 if method == "het_diag" else None),
               "tau": cfg["head"].get("temperature"), "dir": str(run_dir),
               "best_epoch": s["best_epoch"], "num_params": s["num_params"],
               "epoch_time_s": s["mean_epoch_time_s"]}
        for split in ("test_best", "test_last", "test_best_mu", "test_last_mu"):
            for k, v in s.get(split, {}).items():
                row[f"{split}.{k}"] = v
        noise = s.get("noise", {})
        row["noise_rate"] = noise.get("train_rate", 0.0)
        row["auroc"] = noise.get("train_noise_auroc", np.nan)
        corr = s.get("correlation")
        if corr:
            row["pair_precision"] = corr["pair_precision_at_k"]
            row["min_offdiag_corr"] = corr["min_offdiag_corr"]
            for name, v in corr["pair_corr"].items():
                row[f"corr_{name}"] = v
            row["pair_corr_mean"] = np.mean(list(corr["pair_corr"].values()))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Missing columns (e.g. no HET runs yet) become NaN, so partial grids can be aggregated too.
    expected = [f"{split}.{k}" for split in ("test_best", "test_last", "test_best_mu", "test_last_mu")
                for k in ("top1", "top2", "nll", "ece", "pair_top1", "other_top1", "within_pair_error_share")]
    expected += ["pair_precision", "min_offdiag_corr", "pair_corr_mean"] + [f"corr_{i}-{j}" for i, j in PAIRS]
    for col in expected:
        if col not in df:
            df[col] = np.nan
    return df


def ms(series, scale=1.0, digits=2):
    v = series.dropna() * scale
    if v.empty:
        return "-"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{digits}f}"


# ---------------------------------------------------------------- tables

def write_table(df, name, title):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    with open(TABLES / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n{df.to_markdown(index=False)}\n")


def tables(main, abl):
    metrics = [("test_best.top1", "top-1 best [%]", 100, 2), ("test_last.top1", "top-1 last [%]", 100, 2),
               ("test_best.pair_top1", "top-1 in pairs [%]", 100, 2),
               ("test_best.within_pair_error_share", "errors inside pairs [%]", 100, 1),
               ("test_best.nll", "NLL", 1, 3), ("test_best.ece", "ECE [%]", 100, 2),
               ("auroc", "AUROC (mislabeled)", 1, 3)]
    rows = []
    for sc in SCENARIOS:
        for m in METHODS:
            g = main[(main.scenario == sc) & (main.method == m)]
            if g.empty:
                continue
            rows.append({"scenario": SCENARIO_LABEL[sc], "method": METHOD_LABEL[m], "seeds": len(g),
                         **{label: ms(g[col], scale, d) for col, label, scale, d in metrics}})
    write_table(pd.DataFrame(rows), "mnist_main", "MNIST: clean test set, best-val checkpoint (mean ± std over seeds)")

    # Compact pivot: top-1 best per scenario x method.
    pivot = pd.DataFrame({METHOD_LABEL[m]: [ms(main[(main.scenario == sc) & (main.method == m)]["test_best.top1"], 100)
                                            for sc in SCENARIOS] for m in METHODS})
    pivot.insert(0, "scenario", [SCENARIO_LABEL[sc] for sc in SCENARIOS])
    write_table(pivot, "mnist_top1_pivot", "MNIST: clean test top-1 [%], best-val checkpoint")

    # HET prediction modes: MC mean vs softmax(mu).
    rows = []
    for sc in SCENARIOS:
        g = main[(main.scenario == sc) & (main.method == "het")]
        if g.empty:
            continue
        for mode, prefix in (("MC mean", "test_best"), ("softmax(mu)", "test_best_mu")):
            rows.append({"scenario": SCENARIO_LABEL[sc], "HET prediction": mode,
                         "top-1 [%]": ms(g[f"{prefix}.top1"], 100), "NLL": ms(g[f"{prefix}.nll"], 1, 3),
                         "ECE [%]": ms(g[f"{prefix}.ece"], 100)})
    write_table(pd.DataFrame(rows), "mnist_het_modes", "HET: MC-mean vs softmax(mu) prediction on the clean test set")

    # Learned correlation structure.
    rows = []
    for sc in SCENARIOS:
        g = main[(main.scenario == sc) & (main.method == "het")]
        if g.empty:
            continue
        rows.append({"scenario": SCENARIO_LABEL[sc], "precision@3": ms(g["pair_precision"], 1, 2),
                     **{f"corr {i}-{j}": ms(g[f"corr_{i}-{j}"], 1, 3) for i, j in PAIRS},
                     "most negative corr": ms(g["min_offdiag_corr"], 1, 3)})
    write_table(pd.DataFrame(rows), "mnist_correlation",
                "HET: class correlation averaged over test images of class i or j")

    if not abl.empty:
        rows = []
        for (R, tau), g in abl.groupby(["R", "tau"]):
            rows.append({"R": int(R), "tau": tau, "seeds": len(g), "top-1 best [%]": ms(g["test_best.top1"], 100),
                         "top-1 last [%]": ms(g["test_last.top1"], 100), "NLL": ms(g["test_best.nll"], 1, 3),
                         "ECE [%]": ms(g["test_best.ece"], 100), "mean pair corr": ms(g["pair_corr_mean"], 1, 3),
                         "precision@3": ms(g["pair_precision"], 1, 2), "AUROC": ms(g["auroc"], 1, 3)})
        write_table(pd.DataFrame(rows), "mnist_ablation", "Ablation on S1-30%: number of factors R and temperature tau")

    cost = main.groupby("method").agg(params=("num_params", "first"), epoch_s=("epoch_time_s", "mean")).reindex(METHODS)
    cost = cost.reset_index().assign(method=lambda d: d.method.map(METHOD_LABEL))
    cost["epoch_s"] = cost["epoch_s"].round(2)
    write_table(cost, "mnist_cost", "Cost: parameters and mean training time per epoch (GTX 1060)")


# ---------------------------------------------------------------- figures

def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def grouped_bars(ax, main, col, scale, methods, scenarios, ylabel):
    width = 0.8 / len(methods)
    x = np.arange(len(scenarios))
    for k, m in enumerate(methods):
        src_m, src_col = ("het", col.replace("test_best.", "test_best_mu.")) if m == "het_mu" else (m, col)
        means, stds = [], []
        for sc in scenarios:
            v = main[(main.scenario == sc) & (main.method == src_m)][src_col].dropna() * scale
            means.append(v.mean() if len(v) else np.nan)
            stds.append(v.std(ddof=1) if len(v) > 1 else 0)
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, means, width * 0.92, yerr=stds, capsize=2,
               color=COLOR[m], edgecolor="white", linewidth=0.6, hatch="///" if m == "het_mu" else None,
               error_kw={"elinewidth": 0.8, "ecolor": INK}, label=METHOD_LABEL[m])
    ax.set_xticks(x, [SCENARIO_LABEL[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig_accuracy(main):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, title in zip(axes, ["test_best.top1", "test_last.top1"],
                              ["Best-val checkpoint", "Last epoch"]):
        grouped_bars(ax, main, col, 100, METHODS, SCENARIOS, "clean test top-1 [%]")
        ax.set_title(title)
    lo = np.nanmin(main[["test_best.top1", "test_last.top1"]].values) * 100
    axes[0].set_ylim(max(lo - 1.0, 0), 100)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(1.05, -0.42))
    fig.suptitle("MNIST: accuracy on the clean test set (mean ± std over seeds)")
    save(fig, "mnist_top1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, title in zip(axes, ["test_best.pair_top1", "test_best.other_top1"],
                              ["Classes in noisy pairs (1, 7, 4, 9, 3, 8)", "Other classes (0, 2, 5, 6)"]):
        grouped_bars(ax, main, col, 100, METHODS, SCENARIOS, "clean test top-1 [%]")
        ax.set_title(title)
    lo = np.nanmin(main[["test_best.pair_top1", "test_best.other_top1"]].values) * 100
    axes[0].set_ylim(max(lo - 1.0, 0), 100)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(1.05, -0.42))
    fig.suptitle("MNIST: accuracy inside vs outside the confused pairs (best-val checkpoint)")
    save(fig, "mnist_top1_pairs")


def fig_calibration_bars(main):
    methods = METHODS + ["het_mu"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    grouped_bars(axes[0], main, "test_best.nll", 1, methods, SCENARIOS, "NLL (lower is better)")
    grouped_bars(axes[1], main, "test_best.ece", 100, methods, SCENARIOS, "ECE [%] (lower is better)")
    axes[0].set_title("Negative log-likelihood")
    axes[1].set_title("Expected calibration error (15 bins)")
    axes[0].legend(ncol=2)
    fig.suptitle("MNIST: uncertainty quality on the clean test set (best-val checkpoint)")
    save(fig, "mnist_nll_ece")


def fig_auroc(main):
    if main[main.scenario.isin(NOISY)]["auroc"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    grouped_bars(ax, main, "auroc", 1, METHODS, NOISY, "AUROC")
    lo = np.nanmin(main[main.scenario.isin(NOISY)]["auroc"].values)
    ax.set_ylim(max(lo - 0.02, 0.5), 1.0)
    ax.set_title("Finding mislabeled training images: score = -log p(given label | x)")
    ax.legend(ncol=4, loc="lower left")
    save(fig, "mnist_auroc")


def epoch_frame(runs):
    frames = []
    for _, r in runs.iterrows():
        e = pd.read_csv(Path(r["dir"]) / "epochs.csv")
        e["method"], e["seed"] = r["method"], r["seed"]
        frames.append(e)
    return pd.concat(frames)


def fig_curves(main):
    for sc in SCENARIOS:
        runs = main[main.scenario == sc]
        if runs.empty:
            continue
        ep = epoch_frame(runs)
        noisy = sc != "s0"
        panels = [("val_top1", "Val top-1 (noisy val labels)"),
                  ("train_acc_clean", "Train: prediction = true label"),
                  ("train_loss", "Train loss")]
        if noisy:
            panels.insert(2, ("memorized", "Mislabeled train images: predicts wrong label"))
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
        for ax, (col, title) in zip(axes, panels):
            for m in METHODS:
                g = ep[ep.method == m].groupby("epoch")[col]
                if g.ngroups == 0:
                    continue
                mean, std = g.mean(), g.std().fillna(0)
                ax.plot(mean.index, mean.values, color=COLOR[m], lw=2, label=METHOD_LABEL[m])
                ax.fill_between(mean.index, mean - std, mean + std, color=COLOR[m], alpha=0.15, lw=0)
            ax.set_title(title)
            ax.set_xlabel("epoch")
        axes[0].legend()
        fig.suptitle(f"MNIST {SCENARIO_LABEL[sc]}: learning curves (mean ± std over seeds)")
        fig.tight_layout()
        save(fig, f"mnist_curves_{sc}")


def matrix_panel(ax, m, cmap, vmin, vmax, title, fmt=".2f", min_show=0.005):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    K = m.shape[0]
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.grid(False)
    for i in range(K):
        for j in range(K):
            v = m[i, j]
            if not np.isnan(v) and (abs(v) >= min_show or i == j):
                dark = abs(v - (0 if vmin < 0 else vmin)) > 0.55 * (vmax - (0 if vmin < 0 else vmin))
                ax.text(j, i, format(v, fmt).replace("0.", ".", 1) if abs(v) < 1 else format(v, fmt),
                        ha="center", va="center", fontsize=5.5, color="white" if dark else INK)
    ax.set_title(title, fontsize=9)
    return im


def mean_matrix(runs, filename, normalize_rows=False):
    mats = []
    for d in runs["dir"]:
        p = Path(d) / filename
        if p.exists():
            m = np.loadtxt(p, delimiter=",")
            if normalize_rows:
                m = m / m.sum(axis=1, keepdims=True).clip(min=1)
            mats.append(m)
    return np.mean(mats, axis=0) if mats else None


def fig_confusions(main):
    scen = ["s0", "s1_p30", "s1_p45", "s2_p30", "s3_p30"]
    for methods, name in ((METHODS, "mnist_confusion_all"), (["ce", "het"], "mnist_confusion_ce_vs_het")):
        im = None
        fig, axes = plt.subplots(len(methods), len(scen), figsize=(3.0 * len(scen), 3.0 * len(methods)))
        for r, m in enumerate(methods):
            for c, sc in enumerate(scen):
                ax = axes[r, c]
                cm = mean_matrix(main[(main.scenario == sc) & (main.method == m)], "confusion_test.csv", True)
                if cm is None:
                    ax.axis("off")
                    continue
                # Off-diagonal only: the diagonal (~99%) would swamp the color scale.
                off = cm.copy()
                np.fill_diagonal(off, np.nan)
                im = matrix_panel(ax, off * 100, "Blues", 0, 3, f"{METHOD_LABEL[m]}\n{SCENARIO_LABEL[sc]}",
                                  fmt=".1f", min_show=0.25)
                if c == 0:
                    ax.set_ylabel("true class")
                if r == len(methods) - 1:
                    ax.set_xlabel("predicted")
        if im is not None:
            fig.colorbar(im, ax=axes, fraction=0.015, label="% of true class (off-diagonal, color capped at 3%)")
        fig.suptitle("Clean test confusions (row %, mean over seeds); empty cells < 0.25%")
        save(fig, name)


def fig_correlations(main):
    scen = [s for s in SCENARIOS if not main[(main.scenario == s) & (main.method == "het")].empty]
    if not scen:
        return
    fig, axes = plt.subplots(1, len(scen), figsize=(3.1 * len(scen), 3.4))
    for ax, sc in zip(np.atleast_1d(axes), scen):
        corr = mean_matrix(main[(main.scenario == sc) & (main.method == "het")], "correlation_pairs_test.csv")
        np.fill_diagonal(corr, np.nan)
        im = matrix_panel(ax, corr, "RdBu_r", -1, 1, SCENARIO_LABEL[sc], min_show=0.3)
        for i, j in PAIRS:  # mark the true noise pairs
            for a, b in ((i, j), (j, i)):
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    fig.colorbar(im, ax=axes, fraction=0.012, label="correlation")
    fig.suptitle("HET: learned class correlation corr(i, j), avg over test images of class i or j "
                 "(framed: true noise pairs; values |c| < 0.3 hidden)")
    save(fig, "mnist_het_correlation")


def fig_corr_vs_noise(main):
    het = main[(main.method == "het") & main.scenario.isin(["s0", "s1_p10", "s1_p30", "s1_p45"])]
    if het.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    xs, ys = [], []
    pair_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (i, j), col in zip(PAIRS, pair_colors):
        lvl = het.scenario.map({"s0": 0, "s1_p10": 10, "s1_p30": 30, "s1_p45": 45})
        d = pd.DataFrame({"level": lvl, "corr": het[f"corr_{i}-{j}"]}).groupby("level")["corr"]
        ax.errorbar(d.mean().index, d.mean().values, yerr=d.std().fillna(0).values, color=col, marker="o",
                    ms=6, lw=2, capsize=3, label=f"pair {i}-{j}")
        xs += list(lvl.values)
        ys += list(het[f"corr_{i}-{j}"].values)
    rho, p = spearmanr(xs, ys)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([0, 10, 30, 45])
    ax.set_xlabel("pair noise level in S1 [%]")
    ax.set_ylabel("learned correlation")
    ax.set_title(f"HET: correlation vs noise strength (Spearman ρ = {rho:.2f}, p = {p:.1g})", fontsize=10)
    ax.legend()
    save(fig, "mnist_het_corr_vs_noise")


def pooled_reliability(runs, mode):
    frames = [pd.read_csv(Path(d) / "reliability_test.csv") for d in runs["dir"]]
    df = pd.concat(frames)
    df = df[(df["mode"] == mode) & (df["count"] > 0)]
    df = df.assign(cw=df.confidence * df["count"], aw=df.accuracy * df["count"])
    g = df.groupby("bin")[["cw", "aw", "count"]].sum()
    return g.cw / g["count"], g.aw / g["count"]


def fig_reliability(main):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex=True, sharey=True)
    for ax, sc in zip(axes.flat, SCENARIOS):
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
        for m in METHODS + ["het_mu"]:
            runs = main[(main.scenario == sc) & (main.method == ("het" if m == "het_mu" else m))]
            if runs.empty:
                continue
            conf, acc = pooled_reliability(runs, "mu" if m == "het_mu" else "mc")
            ax.plot(conf, acc, color=COLOR[m], marker="o", ms=4, lw=2 if m != "het_mu" else 1.4,
                    ls=":" if m == "het_mu" else "-", label=METHOD_LABEL[m])
        ax.set_title(SCENARIO_LABEL[sc])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("confidence")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Reliability diagrams, clean test (bins pooled over seeds); below diagonal = overconfident")
    fig.tight_layout()
    save(fig, "mnist_reliability")


def fig_noise_transitions(main):
    scen = NOISY
    im = None
    fig, axes = plt.subplots(1, len(scen), figsize=(3.0 * len(scen), 3.2))
    for ax, sc in zip(axes, scen):
        runs = main[main.scenario == sc]
        T = mean_matrix(runs.head(1), "noise_transition.csv") if not runs.empty else None
        if T is None:
            ax.axis("off")
            continue
        im = matrix_panel(ax, T, "Oranges", 0, 1, SCENARIO_LABEL[sc], min_show=0.01)
        ax.set_xlabel("given label")
    axes[0].set_ylabel("true class")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.012, label="P(given label | true class)")
    fig.suptitle("Injected label noise on the training set (identical for all methods and seeds)")
    save(fig, "mnist_noise_transitions")


def fig_ablation(abl):
    if abl.empty:
        return
    Rs, taus = sorted(abl.R.unique()), sorted(abl.tau.unique())
    specs = [("test_best.top1", "clean test top-1 [%]", 100, "Blues", ".2f"),
             ("test_best.nll", "clean test NLL", 1, "Blues_r", ".3f"),
             ("pair_corr_mean", "mean pair correlation", 1, "RdBu_r", ".2f")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (col, title, scale, cmap, fmt) in zip(axes, specs):
        M = np.array([[abl[(abl.R == R) & (abl.tau == t)][col].mean() * scale for t in taus] for R in Rs])
        kw = {"vmin": -1, "vmax": 1} if cmap == "RdBu_r" else {}
        im = ax.imshow(M, cmap=cmap, aspect="auto", **kw)
        for a in range(len(Rs)):
            for b in range(len(taus)):
                ax.text(b, a, format(M[a, b], fmt), ha="center", va="center", fontsize=8,
                        color="white" if im.norm(M[a, b]) > 0.6 or im.norm(M[a, b]) < 0.15 and cmap == "RdBu_r"
                        else INK)
        ax.set_xticks(range(len(taus)), [f"{t:g}" for t in taus])
        ax.set_yticks(range(len(Rs)), [str(int(r)) for r in Rs])
        ax.set_xlabel("temperature τ")
        ax.set_ylabel("factors R")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("HET ablation on S1-30% (mean over seeds)")
    fig.tight_layout()
    save(fig, "mnist_ablation")


def main():
    df = load_runs()
    if df.empty:
        print("no runs found")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "mnist_runs.csv", index=False)
    main_runs = df[~df.ablation]
    het_main_s1 = main_runs[(main_runs.scenario == "s1_p30") & (main_runs.method == "het")]
    abl = pd.concat([df[df.ablation], het_main_s1])
    tables(main_runs, abl)
    fig_accuracy(main_runs)
    fig_calibration_bars(main_runs)
    fig_auroc(main_runs)
    fig_curves(main_runs)
    fig_confusions(main_runs)
    fig_correlations(main_runs)
    fig_corr_vs_noise(main_runs)
    fig_reliability(main_runs)
    fig_noise_transitions(main_runs)
    fig_ablation(abl)
    print(f"{len(df)} runs -> {TABLES}, {FIGURES}")


if __name__ == "__main__":
    main()
