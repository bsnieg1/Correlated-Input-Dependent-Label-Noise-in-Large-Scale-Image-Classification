"""Report for the Multispecies Campus UJ experiments (results/uj_*/seed*/).

    python -m analysis.uj_report

Step 1 = trained on SpeciesNet labels, step 2 = trained on expert labels; the test set always uses expert
labels and contains only photo sequences never seen in training. Writes results/tables/uj_*.{csv,md} and
results/figures/uj_*.png. All numbers are mean +- std over seeds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from analysis.aggregate import COLOR, FIGURES, INK, METHOD_LABEL, METHODS, MUTED, RESULTS, matrix_panel, ms, plt, save, \
    write_table
from src.metrics import auroc

METHOD_LABEL = {**METHOD_LABEL, "het": "HET (R=5)"}  # UJ configs use R=5 (MNIST main grid: R=3)
from src.uj_data import CLASSES, CROP_BOTTOM, build_table

STEPS = {"model": "Step 1: SpeciesNet labels", "expert": "Step 2: expert labels"}
SHORT = {"wild boar": "boar", "european roe deer": "roe deer", "european hare": "hare", "domestic cat": "cat",
         "ring-necked pheasant": "pheasant", "eurasian badger": "badger", "common blackbird": "blackbird",
         "red fox": "fox", "red deer": "red deer", "common woodpigeon": "woodpigeon", "marten": "marten",
         "song thrush": "song thrush", "domestic dog": "dog", "eurasian red squirrel": "red squirrel",
         "other_bird": "other bird", "other_mammal": "other mammal", "blank": "blank"}
NAMES = [SHORT[c] for c in CLASSES]


def load():
    runs = []
    for p in sorted(RESULTS.glob("uj_*/seed*/summary.json")):
        s = json.loads(p.read_text())
        cfg = yaml.safe_load((p.parent / "config.yaml").read_text())
        runs.append({"dir": p.parent, "step": cfg["data"]["train_labels"], "method": cfg["method"],
                     "seed": s["seed"], "summary": s, "pred": np.load(p.parent / "predictions.npz")})
    cfg = yaml.safe_load((runs[0]["dir"] / "config.yaml").read_text())
    meta = build_table(cfg["data"]).set_index("id")
    return runs, meta


def speciesnet_row(meta):
    t = meta[meta.split == "test"]
    pred = torch.nn.functional.one_hot(torch.tensor(t.y_model.values), len(CLASSES)).float().add(1e-6).log()
    from src.metrics import balanced_accuracy
    return {"top1": (t.y_model == t.y_expert).mean(), "balanced_acc": balanced_accuracy(pred, torch.tensor(t.y_expert.values))}


def main_table(runs, meta):
    sn = speciesnet_row(meta)
    rows = [{"training labels": "-", "method": "SpeciesNet itself (no training)", "seeds": "-",
             "top-1 best [%]": f"{sn['top1'] * 100:.2f}", "top-1 last [%]": "-",
             "balanced acc [%]": f"{sn['balanced_acc'] * 100:.2f}", "NLL": "-", "ECE [%]": "-"}]
    for step in STEPS:
        for m in METHODS:
            g = [r["summary"] for r in runs if r["step"] == step and r["method"] == m]
            if not g:
                continue
            col = lambda split, k: pd.Series([s[split][k] for s in g])
            rows.append({"training labels": STEPS[step], "method": METHOD_LABEL[m], "seeds": len(g),
                         "top-1 best [%]": ms(col("test_best", "top1"), 100), "top-1 last [%]": ms(col("test_last", "top1"), 100),
                         "balanced acc [%]": ms(col("test_best", "balanced_acc"), 100),
                         "NLL": ms(col("test_best", "nll"), 1, 3), "ECE [%]": ms(col("test_best", "ece"), 100)})
            if m == "het":
                rows.append({"training labels": STEPS[step], "method": METHOD_LABEL["het_mu"], "seeds": len(g),
                             "top-1 best [%]": ms(col("test_best_mu", "top1"), 100), "top-1 last [%]": "-",
                             "balanced acc [%]": ms(col("test_best_mu", "balanced_acc"), 100),
                             "NLL": ms(col("test_best_mu", "nll"), 1, 3), "ECE [%]": ms(col("test_best_mu", "ece"), 100)})
    write_table(pd.DataFrame(rows), "uj_main",
                "Multispecies Campus UJ: clean test set (expert labels, unseen photo sequences)")
    return sn


def test_frame(run, meta):
    p = run["pred"]
    t = meta.loc[p["test_ids"]]
    logp = p["test_logp"]
    y_model, y_exp = t.y_model.values, t.y_expert.values
    return pd.DataFrame({"id": p["test_ids"], "y_model": y_model, "y_expert": y_exp, "pred": logp.argmax(1),
                         "score_model_label": -logp[np.arange(len(t)), y_model],
                         "score_expert_label": -logp[np.arange(len(t)), y_exp],
                         "speciesnet_conf": t.scores.values})


def detection_tables(runs, meta):
    """Finding SpeciesNet mistakes on unseen test images, and how many of them the network fixes."""
    rows, auroc_rows = [], []
    t0 = test_frame(runs[0], meta)
    wrong = torch.tensor((t0.y_model != t0.y_expert).values)
    sn_auroc = auroc(torch.tensor(1 - t0.speciesnet_conf.values), wrong)
    for step in STEPS:
        for m in METHODS:
            g = [test_frame(r, meta) for r in runs if r["step"] == step and r["method"] == m]
            if not g:
                continue
            au = pd.Series([auroc(torch.tensor(f.score_model_label.values), wrong) for f in g])
            fixed = pd.Series([((f.pred == f.y_expert) & (f.y_model != f.y_expert)).sum() / (f.y_model != f.y_expert).sum()
                               for f in g])
            broke = pd.Series([((f.pred != f.y_expert) & (f.y_model == f.y_expert)).sum() / (f.y_model == f.y_expert).sum()
                               for f in g])
            rows.append({"training labels": STEPS[step], "method": METHOD_LABEL[m],
                         "AUROC finding SpeciesNet errors": ms(au, 1, 3),
                         "SpeciesNet errors fixed [%]": ms(fixed, 100, 1),
                         "correct SpeciesNet labels broken [%]": ms(broke, 100, 1)})
            auroc_rows.append((step, m, au.mean(), au.std(ddof=1) if len(au) > 1 else 0))
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"training labels": "-", "method": "SpeciesNet confidence (1 - score)",
                       "AUROC finding SpeciesNet errors": f"{sn_auroc:.3f}", "SpeciesNet errors fixed [%]": "-",
                       "correct SpeciesNet labels broken [%]": "-"}
    write_table(df, "uj_error_detection",
                f"Finding and fixing SpeciesNet mistakes on the test set ({int(wrong.sum())} of {len(wrong)} labels wrong)")
    return auroc_rows, sn_auroc


def fig_accuracy(runs, sn):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, key, title, ref in zip(axes, ["top1", "balanced_acc"], ["Top-1 accuracy", "Balanced accuracy (mean per-class recall)"],
                                   [sn["top1"], sn["balanced_acc"]]):
        width = 0.8 / len(METHODS)
        for k, m in enumerate(METHODS):
            means, stds = [], []
            for step in STEPS:
                v = pd.Series([r["summary"]["test_best"][key] for r in runs if r["step"] == step and r["method"] == m]) * 100
                means.append(v.mean()); stds.append(v.std(ddof=1) if len(v) > 1 else 0)
            ax.bar(np.arange(len(STEPS)) + (k - 1.5) * width, means, width * 0.92, yerr=stds, capsize=2, color=COLOR[m],
                   edgecolor="white", error_kw={"elinewidth": 0.8, "ecolor": INK}, label=METHOD_LABEL[m])
        ax.axhline(ref * 100, color=INK, ls="--", lw=1.2, label="SpeciesNet itself")
        ax.set_xticks(range(len(STEPS)), list(STEPS.values()))
        ax.set_ylabel("clean test [%]")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        lo = min(ref * 100, min(r["summary"]["test_best"][key] for r in runs) * 100)
        ax.set_ylim(max(lo - 8, 0), None)
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(1.05, -0.32))
    fig.suptitle("Multispecies Campus UJ: ResNet-18 (ImageNet) fine-tuned, test on expert labels (mean ± std, 3 seeds)")
    save(fig, "uj_accuracy")


def fig_auroc(auroc_rows, sn_auroc):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.8 / len(METHODS)
    for k, m in enumerate(METHODS):
        vals = [(mu, sd) for st in STEPS for (s, mm, mu, sd) in auroc_rows if s == st and mm == m]
        ax.bar(np.arange(len(STEPS)) + (k - 1.5) * width, [v[0] for v in vals], width * 0.92,
               yerr=[v[1] for v in vals], capsize=2, color=COLOR[m], edgecolor="white", label=METHOD_LABEL[m],
               error_kw={"elinewidth": 0.8, "ecolor": INK})
    ax.axhline(sn_auroc, color=INK, ls="--", lw=1.2, label="SpeciesNet confidence")
    ax.set_xticks(range(len(STEPS)), list(STEPS.values()))
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC")
    ax.grid(axis="x", visible=False)
    ax.set_title("Finding SpeciesNet mistakes on unseen test images: score = -log p(SpeciesNet label | x)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    save(fig, "uj_error_detection_auroc")


def mean_matrix(runs, filename, normalize_rows=False):
    mats = []
    for r in runs:
        m = np.loadtxt(r["dir"] / filename, delimiter=",")
        if normalize_rows:
            m = m / m.sum(axis=1, keepdims=True).clip(min=1)
        mats.append(m)
    return np.mean(mats, axis=0)


def fig_confusion_and_correlation(runs, meta):
    present = [i for i, c in enumerate(CLASSES) if c != "blank"]
    names = [NAMES[i] for i in present]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, (step, m) in zip(axes[:2], [("model", "ce"), ("model", "het")]):
        cm = mean_matrix([r for r in runs if r["step"] == step and r["method"] == m], "confusion_test.csv", True)
        cm = cm[np.ix_(present, range(len(CLASSES)))][:, present] * 100
        matrix_panel(ax, cm, "Blues", 0, 100, f"{METHOD_LABEL[m]}, {STEPS[step]}\nclean test confusion (row %)", fmt=".0f",
                     min_show=4.5)
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("expert label")
    het = [r for r in runs if r["step"] == "model" and r["method"] == "het"]
    corr = mean_matrix(het, "correlation_pairs_test.csv")
    np.fill_diagonal(corr, np.nan)
    im = matrix_panel(axes[2], corr, "RdBu_r", -1, 1, "HET (step 1): learned class correlation\n(framed: most frequent SpeciesNet swaps)",
                      min_show=0.4)
    axes[2].set_xticks(range(len(NAMES)), NAMES, rotation=60, ha="right")
    axes[2].set_yticks(range(len(NAMES)), NAMES)
    pairs = json.loads((het[0]["dir"] / "summary.json").read_text())["correlation"]["pair_corr"]
    for key in pairs:
        i, j = map(int, key.split("-"))
        for a, b in ((i, j), (j, i)):
            axes[2].add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    save(fig, "uj_confusion_correlation")


def fig_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, step in zip(axes, STEPS):
        for m in METHODS:
            eps = [pd.read_csv(r["dir"] / "epochs.csv") for r in runs if r["step"] == step and r["method"] == m]
            e = pd.concat(eps).groupby("epoch")["val_top1"]
            ax.plot(e.mean().index, e.mean().values, color=COLOR[m], lw=2, label=METHOD_LABEL[m])
            ax.fill_between(e.mean().index, e.mean() - e.std(), e.mean() + e.std(), color=COLOR[m], alpha=0.15, lw=0)
        ax.set_title(f"{STEPS[step]}\nval top-1 (val labels = training labels)", fontsize=10)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("val top-1")
    axes[0].legend()
    save(fig, "uj_curves")


def gallery(frame, meta, images_dir, path, title, caption):
    n = len(frame)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows))
    for ax, (_, r) in zip(axes.flat, frame.iterrows()):
        row = meta.loc[r.id]
        im = Image.open(Path(images_dir) / row.folder / row["name"])
        w, h = im.size
        ax.imshow(im.crop((0, 0, w, int(h * (1 - CROP_BOTTOM)))))
        ax.set_title(caption(r, row), fontsize=8, color=INK)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def suspects(runs, meta, images_dir):
    """Step 2: test images whose expert label the HET models find least plausible (for manual re-checking).
    Step 1: test images flagged as SpeciesNet mistakes (does the flag hit real mistakes?)."""
    out = {}
    for step, label_col, score_col in (("expert", "y_expert", "score_expert_label"), ("model", "y_model", "score_model_label")):
        frames = [test_frame(r, meta) for r in runs if r["step"] == step and r["method"] == "het"]
        f = frames[0][["id", "y_model", "y_expert"]].copy()
        f["score"] = np.mean([fr[score_col].values for fr in frames], axis=0)
        # Majority vote over seeds; on a three-way tie mode() is sorted, so take its first row.
        f["pred"] = pd.DataFrame([fr.pred.values for fr in frames]).mode().iloc[0].astype(int).values
        f = f.sort_values("score", ascending=False)
        out[step] = f
    exp = out["expert"].head(40).assign(
        expert_label=lambda d: [CLASSES[i] for i in d.y_expert], speciesnet_label=lambda d: [CLASSES[i] for i in d.y_model],
        network_prediction=lambda d: [CLASSES[i] for i in d.pred])
    exp[["id", "score", "expert_label", "network_prediction", "speciesnet_label"]].to_csv(
        RESULTS / "tables" / "uj_suspected_expert_labels.csv", index=False)
    gallery(out["expert"].head(16), meta, images_dir, FIGURES / "uj_suspected_expert_labels.png",
            "Step 2 (HET trained on expert labels): 16 test photos whose expert label the network finds least plausible",
            lambda r, row: f"expert: {SHORT[CLASSES[r.y_expert]]} | net: {SHORT[CLASSES[r.pred]]}\n{r.id}")
    top = out["model"].head(16)
    hit = (top.y_model != top.y_expert).mean()
    gallery(top, meta, images_dir, FIGURES / "uj_flagged_speciesnet_labels.png",
            f"Step 1 (HET trained on SpeciesNet labels): 16 most suspicious SpeciesNet labels on test photos "
            f"- {hit:.0%} are real mistakes",
            lambda r, row: (f"SpeciesNet: {SHORT[CLASSES[r.y_model]]} -> expert: {SHORT[CLASSES[r.y_expert]]}"
                            f" {'(mistake)' if r.y_model != r.y_expert else '(correct)'}\nnet: {SHORT[CLASSES[r.pred]]} | {r.id}"))
    k = 50
    prec = (out["model"].head(k).y_model != out["model"].head(k).y_expert).mean()
    base = (out["model"].y_model != out["model"].y_expert).mean()
    return {"precision_at_50": prec, "base_rate": base}


def main():
    runs, meta = load()
    images_dir = yaml.safe_load((runs[0]["dir"] / "config.yaml").read_text())["data"]["images_dir"]
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    sn = main_table(runs, meta)
    auroc_rows, sn_auroc = detection_tables(runs, meta)
    fig_accuracy(runs, sn)
    fig_auroc(auroc_rows, sn_auroc)
    fig_confusion_and_correlation(runs, meta)
    fig_curves(runs)
    s = suspects(runs, meta, images_dir)
    t = meta[meta.split == "test"]
    info = {"test_images": len(t), "speciesnet_wrong_on_test": int((t.y_model != t.y_expert).sum()),
            "het_step1_precision_at_50": s["precision_at_50"], "base_rate": s["base_rate"], "runs": len(runs)}
    (RESULTS / "tables" / "uj_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
