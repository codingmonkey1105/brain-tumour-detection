"""ImageDataGenerator pipeline (rescale + training-only augmentation).

Reads the manifests produced by src/data.py. The same split files feed every
model, so all comparisons use identical train/val/test images.
"""
import json
from pathlib import Path

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def make_generators(data_dir="data", img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42):
    """Return (train_gen, val_gen, test_gen).

    * Training: rescale to [0, 1] + rotation 15 deg, shifts 10%, zoom 10%, horizontal flip.
    * Validation / test: rescale only (no augmentation), no shuffling.
    """
    d = Path(data_dir)
    train_df = pd.read_csv(d / "train.csv")
    val_df = pd.read_csv(d / "val.csv")
    test_df = pd.read_csv(d / "test.csv")

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    plain = ImageDataGenerator(rescale=1.0 / 255)

    def flow(gen, df, shuffle):
        return gen.flow_from_dataframe(
            df.assign(label=df["label"].astype("float32")),
            x_col="filepath",
            y_col="label",
            class_mode="raw",          # numeric 0/1 labels for a sigmoid output
            target_size=img_size,
            color_mode="rgb",          # grayscale MRI slices -> 3 channels
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    return flow(train_aug, train_df, True), flow(plain, val_df, False), flow(plain, test_df, False)


def load_class_weight(data_dir="data"):
    """Class weights inversely proportional to class frequency (from data_report.json)."""
    rep = json.loads((Path(data_dir) / "data_report.json").read_text())
    return {int(k): float(v) for k, v in rep["class_weight"].items()}
