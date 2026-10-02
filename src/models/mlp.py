"""Non-convolutional MLP baseline: 64x64 grayscale images, flattened."""
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

MLP_SIZE = (64, 64)


def load_flat(csv_path):
    """Load a manifest as (X, y): grayscale, 64x64, scaled to [0, 1], flattened."""
    df = pd.read_csv(csv_path)
    X = np.stack(
        [
            np.asarray(Image.open(p).convert("L").resize(MLP_SIZE), dtype="float32").ravel() / 255.0
            for p in df["filepath"]
        ]
    )
    return X, df["label"].to_numpy(dtype="float32")


def build_mlp(input_dim=MLP_SIZE[0] * MLP_SIZE[1], lr=1e-3, dropout=0.4):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="mlp_baseline",
    )
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model
