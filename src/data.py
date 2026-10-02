"""Dataset preparation: verification, binary merge, duplicate check, split.

Dataset: Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle).
Expected layout under --data-dir:
    Training/{glioma,meningioma,pituitary,notumor}/*.jpg
    Testing/{glioma,meningioma,pituitary,notumor}/*.jpg

Output (written to --out-dir):
    train.csv, val.csv, test.csv   columns: filepath, label (0/1), class_name
    data_report.json               counts, dropped files, duplicate statistics
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

TUMOR_CLASSES = ("glioma", "meningioma", "pituitary")
NO_TUMOR_CLASS = "notumor"
IMG_EXT = {".jpg", ".jpeg", ".png"}
SEED = 42


def is_readable(path: Path) -> bool:
    """Open and verify an image; unreadable files are dropped."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_split(root: Path, split: str):
    """Return (DataFrame of readable images, list of unreadable paths)."""
    rows, bad = [], []
    for cls_dir in sorted((root / split).iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            if not is_readable(p):
                bad.append(str(p))
                continue
            rows.append(
                {
                    "filepath": str(p),
                    "class_name": cls_dir.name,
                    # binary merge: glioma/meningioma/pituitary -> 1, notumor -> 0
                    "label": 0 if cls_dir.name == NO_TUMOR_CLASS else 1,
                    "split_folder": split,
                }
            )
    return pd.DataFrame(rows), bad


def main(data_dir: str, out_dir: str, val_size: float, seed: int, drop_duplicates: bool):
    root, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_all, bad_tr = scan_split(root, "Training")
    test_all, bad_te = scan_split(root, "Testing")

    # ---- exact-duplicate check (file hashing) across Training and Testing ----
    train_all["hash"] = train_all["filepath"].map(lambda p: md5(Path(p)))
    test_all["hash"] = test_all["filepath"].map(lambda p: md5(Path(p)))

    dup_within_train = int(train_all.duplicated("hash", keep="first").sum())
    dup_within_test = int(test_all.duplicated("hash", keep="first").sum())
    train_in_test_mask = train_all["hash"].isin(set(test_all["hash"]))
    leak_train_test = int(train_in_test_mask.sum())

    if drop_duplicates:
        # keep the Testing folder untouched; remove leaking / repeated Training files
        train_all = train_all[~train_in_test_mask]
        train_all = train_all.drop_duplicates("hash", keep="first")

    # ---- stratified, seeded train/val split (20% val) ----
    # Stratify on the 4-class name so both binary and per-tumor-type ratios are kept.
    train_df, val_df = train_test_split(
        train_all,
        test_size=val_size,
        stratify=train_all["class_name"],
        random_state=seed,
    )

    cols = ["filepath", "label", "class_name"]
    train_df[cols].to_csv(out / "train.csv", index=False)
    val_df[cols].to_csv(out / "val.csv", index=False)
    test_all[cols].to_csv(out / "test.csv", index=False)

    def counts(df):
        return {
            "n": int(len(df)),
            "tumor": int((df.label == 1).sum()),
            "no_tumor": int((df.label == 0).sum()),
        }

    report = {
        "seed": seed,
        "val_size": val_size,
        "unreadable_files": {"Training": bad_tr, "Testing": bad_te},
        "duplicates": {
            "within_training": dup_within_train,
            "within_testing": dup_within_test,
            "training_files_also_in_testing": leak_train_test,
            "dropped_from_training": drop_duplicates,
        },
        "train": counts(train_df),
        "val": counts(val_df),
        "test": counts(test_all),
    }
    n_tumor, n_no = report["train"]["tumor"], report["train"]["no_tumor"]
    n = n_tumor + n_no
    # class weights inversely proportional to class frequency: n / (2 * n_c)
    report["class_weight"] = {"0": n / (2 * n_no), "1": n / (2 * n_tumor)}
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--keep-duplicates", action="store_true", help="do not drop duplicate files")
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, a.val_size, a.seed, not a.keep_duplicates)
