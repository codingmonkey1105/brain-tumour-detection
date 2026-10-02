"""Train one model on the common split and evaluate on validation (and optionally test).

Usage:
    python -m src.train --model vgg16
    python -m src.train --model resnet50 --seed 1
    python -m src.train --model mlp
    python -m src.train --model hybrid          # in progress
Test-set evaluation is reserved for the final report: add --final-test only then.
"""
import argparse

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.evaluate import evaluate_arrays, evaluate_generator
from src.generators import load_class_weight, make_generators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "vgg16", "resnet50", "hybrid"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--final-test", action="store_true", help="also evaluate on the held-out test set")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(a.seed)
    cw = load_class_weight(a.data_dir)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    name = f"{a.model}_seed{a.seed}"

    if a.model == "mlp":
        from src.models.mlp import build_mlp, load_flat

        Xtr, ytr = load_flat(f"{a.data_dir}/train.csv")
        Xva, yva = load_flat(f"{a.data_dir}/val.csv")
        model = build_mlp()
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=32,
                  class_weight=cw, callbacks=[es])
        evaluate_arrays(model, Xva, yva, f"{name}_val")
        if a.final_test:
            Xte, yte = load_flat(f"{a.data_dir}/test.csv")
            evaluate_arrays(model, Xte, yte, f"{name}_test")
        return

    if a.model == "vgg16":
        from src.models.vgg16 import build_vgg16 as build
    elif a.model == "resnet50":
        from src.models.resnet50 import build_resnet50 as build
    else:
        from src.models.hybrid import build_hybrid as build

    train_gen, val_gen, test_gen = make_generators(a.data_dir, seed=a.seed)
    model = build()
    model.fit(train_gen, validation_data=val_gen, epochs=a.epochs, class_weight=cw, callbacks=[es])
    evaluate_generator(model, val_gen, pd.read_csv(f"{a.data_dir}/val.csv")["label"], f"{name}_val")
    if a.final_test:
        evaluate_generator(model, test_gen, pd.read_csv(f"{a.data_dir}/test.csv")["label"], f"{name}_test")


if __name__ == "__main__":
    main()
