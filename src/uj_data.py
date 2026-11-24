"""Multispecies Campus UJ camera-trap dataset (Zenodo 20425725) with two label sources.

    model labels : SpeciesNet predictions ("predicted/results_to_validate_*.xlsx")
    expert labels: the same rows corrected by people ("true/results_VALIDATED_*.xlsx")

Both are mapped to one class set (CLASSES). Expert names get spelling/synonym fixes only. Model names are
mapped mechanically through the SpeciesNet taxonomy (no expert knowledge): a prediction becomes a target
species only if it is that species (or, for "marten", any Martes species); other birds/mammals go to
other_bird/other_mammal; blank/human/vehicle go to blank.

Images (downscaled by cloud/prepare_images.py) are cropped (camera info bar at the bottom), resized and
cached as uint8 tensors. The split is by photo sequence (same camera, gaps <= SEQUENCE_GAP_S), stratified
by expert class, so bursts of one animal never end up in both train and test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["wild boar", "european roe deer", "european hare", "domestic cat", "ring-necked pheasant",
           "eurasian badger", "common blackbird", "red fox", "red deer", "common woodpigeon", "marten",
           "song thrush", "domestic dog", "eurasian red squirrel", "other_bird", "other_mammal", "blank"]
# (genus, species) of each target species in the SpeciesNet taxonomy; species None = any species of the genus.
TARGET_TAXA = {
    ("sus", "scrofa"): "wild boar", ("capreolus", "capreolus"): "european roe deer",
    ("lepus", "europaeus"): "european hare", ("felis", "catus"): "domestic cat",
    ("phasianus", "colchicus"): "ring-necked pheasant", ("meles", "meles"): "eurasian badger",
    ("turdus", "merula"): "common blackbird", ("vulpes", "vulpes"): "red fox", ("cervus", "elaphus"): "red deer",
    ("columba", "palumbus"): "common woodpigeon", ("martes", None): "marten",
    ("turdus", "philomelos"): "song thrush", ("canis", "familiaris"): "domestic dog",
    ("sciurus", "vulgaris"): "eurasian red squirrel",
}
# Spelling variants and shorthands used by the human annotators (ground truth side only).
EXPERT_FIXES = {"raccon dog": "raccoon dog", "blackbird": "common blackbird", "european badger": "eurasian badger",
                "red squirrel": "eurasian red squirrel", "great tit": "bird", "hedgehog species": "mammal"}
SEQUENCE_GAP_S = 120
CROP_BOTTOM = 0.08  # camera info bar (Bushnell / DSCF / BOLY) is at most ~7% of the height
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def load_taxonomy(path):
    tax = pd.read_csv(path, sep=";", header=None, dtype=str, keep_default_na=False,
                      names=["uuid", "cls", "order", "family", "genus", "species", "name"])
    return tax.drop_duplicates("name").set_index("name")


def map_name(name, tax):
    """Common name -> class in CLASSES via the taxonomy; None if the name is unknown."""
    if name not in tax.index:
        return None
    t = tax.loc[name]
    if t.genus and t.species and (t.genus, t.species) in TARGET_TAXA:
        return TARGET_TAXA[(t.genus, t.species)]
    if t.genus and (t.genus, None) in TARGET_TAXA:
        return TARGET_TAXA[(t.genus, None)]
    if t.cls == "aves":
        return "other_bird"
    if t.cls == "mammalia" and t.genus != "homo":
        return "other_mammal"
    return "blank"  # blank, human, vehicle, no taxonomy


def read_label_files(labels_dir):
    """All predicted/validated Excel pairs -> one row per photo with model and expert labels."""
    rows = []
    for p in sorted(Path(labels_dir).glob("*/*/predicted/*.xlsx")):
        t = p.parent.parent / "true" / p.name.replace("results_to_validate_", "results_VALIDATED_")
        dp = pd.read_excel(p)[["Filepath", "Classification", "scores"]]
        dt = pd.read_excel(t)[["Filepath", "Classification"]]
        rows.append(dp.merge(dt, on="Filepath", suffixes=("_model", "_expert")))
    df = pd.concat(rows, ignore_index=True)
    parts = df.Filepath.str.split("/")
    df["folder"], df["name"] = parts.str[1], parts.str[-1]
    for col in ("Classification_model", "Classification_expert"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def build_table(cfg):
    """Labels joined with the downscaled images; adds class ids, sequences and the split."""
    tax = load_taxonomy(cfg["taxonomy"])
    labels = read_label_files(cfg["labels_dir"])
    manifest = pd.read_csv(Path(cfg["images_dir"]) / "manifest.csv")
    df = manifest.merge(labels, on=["folder", "name"], how="inner")
    expert = df.Classification_expert.replace(EXPERT_FIXES)
    df["expert"] = expert.map(lambda n: map_name(n, tax))
    df["model"] = df.Classification_model.map(lambda n: map_name(n, tax))
    unknown = sorted(set(expert[df.expert.isna()]) | set(df.Classification_model[df.model.isna()]))
    if unknown:
        raise ValueError(f"Names missing from the taxonomy: {unknown}")
    df = df[df.expert != "blank"].reset_index(drop=True)  # Zenodo has animal photos only
    idx = {c: i for i, c in enumerate(CLASSES)}
    df["y_expert"], df["y_model"] = df.expert.map(idx), df.model.map(idx)

    # Sequences: same camera folder, consecutive EXIF timestamps closer than SEQUENCE_GAP_S.
    df["time"] = pd.to_datetime(df.exif_datetime, format="%Y:%m:%d %H:%M:%S")
    df = df.sort_values(["folder", "time", "name"]).reset_index(drop=True)
    new_seq = (df.folder != df.folder.shift()) | (df.time.diff().dt.total_seconds() > SEQUENCE_GAP_S)
    df["sequence"] = new_seq.cumsum()

    folds = StratifiedGroupKFold(n_splits=cfg.get("n_folds", 7), shuffle=True, random_state=cfg.get("split_seed", 0))
    fold = np.empty(len(df), dtype=int)
    for k, (_, test_idx) in enumerate(folds.split(df, df.y_expert, groups=df.sequence)):
        fold[test_idx] = k
    df["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    df["id"] = df.folder + "/" + df.name
    return df


def load_images(df, images_dir, size, cache_path):
    """uint8 tensor [N, 3, H, W]: bottom info bar cropped, resized to `size` (no aspect-preserving pad)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path)
        if cached["ids"] == df.id.tolist():
            return cached["x"]
    H, W = size
    x = torch.empty(len(df), 3, H, W, dtype=torch.uint8)
    for i, (folder, name) in enumerate(zip(df.folder, df.name)):
        im = Image.open(Path(images_dir) / folder / name).convert("RGB")
        w, h = im.size
        im = im.crop((0, 0, w, int(round(h * (1 - CROP_BOTTOM))))).resize((W, H), Image.BILINEAR)
        x[i] = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": df.id.tolist(), "x": x}, cache_path)
    return x


def make_transforms(crop):
    """Train: random crop + horizontal flip; eval: center crop. Both return float tensors normalized for ImageNet."""
    ch, cw = crop

    def normalize(x):
        return (x.float() - MEAN.to(x.device)) / STD.to(x.device)

    def train_tf(x):
        H, W = x.shape[-2:]
        i = torch.randint(0, H - ch + 1, (1,)).item()
        j = torch.randint(0, W - cw + 1, (1,)).item()
        x = x[..., i:i + ch, j:j + cw]
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return normalize(x)

    def eval_tf(x):
        H, W = x.shape[-2:]
        i, j = (H - ch) // 2, (W - cw) // 2
        return normalize(x[..., i:i + ch, j:j + cw])

    return train_tf, eval_tf


def top_confused_pairs(y_given, y_true, k):
    """The k unordered class pairs most often swapped between given and true labels."""
    K = len(CLASSES)
    T = torch.bincount(y_true * K + y_given, minlength=K * K).view(K, K)
    S = (T + T.T).triu(1)
    flat = S.flatten().argsort(descending=True)[:k]
    return [(int(f) // K, int(f) % K) for f in flat if S.flatten()[f] > 0]


def load_uj(cfg, device):
    """Returns (splits, noise_stats) in the format of src.data.load_dataset, plus extra keys:
    transforms, class_names, pairs, ids, meta (the full per-image table)."""
    df = build_table(cfg)
    size, crop = tuple(cfg.get("size", (240, 448))), tuple(cfg.get("crop", (224, 416)))
    x_all = load_images(df, cfg["images_dir"], size, Path(cfg["cache_dir"]) / f"uj_{size[0]}x{size[1]}.pt")
    train_labels = cfg["train_labels"]  # "model" (step 1) or "expert" (step 2)
    y_exp = torch.tensor(df.y_expert.values)
    y_given = torch.tensor(df[f"y_{train_labels}"].values)

    def split(name, test=False):
        m = torch.tensor((df.split == name).values)
        y = y_exp[m] if test else y_given[m]
        return x_all[m], y.to(device), y_exp[m].to(device)  # images stay on the CPU (uint8)

    splits = {"train": split("train"), "val": split("val"), "test": split("test", test=True)}
    tr_given, tr_true = splits["train"][1], splits["train"][2]
    # Reference pairs = the most frequent model/expert swaps on the training set (same for both steps).
    m_train = torch.tensor((df.split == "train").values)
    pairs = top_confused_pairs(torch.tensor(df.y_model.values)[m_train], y_exp[m_train], cfg.get("num_pairs", 5))
    train_tf, eval_tf = make_transforms(crop)
    splits.update(transforms={"train": train_tf, "eval": eval_tf}, class_names=CLASSES, pairs=pairs,
                  ids={s: df.id[df.split == s].tolist() for s in ("train", "val", "test")}, meta=df)
    noise_stats = None
    if train_labels == "model":
        noise_stats = {"noise_rate": (tr_given != tr_true).float().mean().item()}
    return splits, noise_stats
