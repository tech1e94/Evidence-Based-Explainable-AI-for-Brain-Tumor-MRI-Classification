"""Train the baseline (original notebook) or the improved model and save it to ``models/``.

    python train.py --data archive --arch improved --epochs 30
    python train.py --data archive --arch baseline --epochs 10      # reproduces the notebook model

Uses the same deterministic validation split as the XAI scripts (last 10 % of every class).
"""
import argparse
import json
import os

from xai_brain.config import CATEGORIES
from xai_brain.data import find_split_dir, list_images, make_tf_dataset, train_val_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="archive")
    ap.add_argument("--arch", choices=["baseline", "improved"], default="improved")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    from tensorflow import keras
    from xai_brain.model import build_baseline_model, build_improved_model

    keras.utils.set_random_seed(0)
    paths, labels = list_images(find_split_dir(a.data, "Training"))
    (tr_p, tr_y), (va_p, va_y) = train_val_split(paths, labels, 0.1)
    te_p, te_y = list_images(find_split_dir(a.data, "Testing"))
    train = make_tf_dataset(tr_p, tr_y, a.batch, augment=(a.arch == "improved"), shuffle=True)
    val = make_tf_dataset(va_p, va_y, a.batch)
    test = make_tf_dataset(te_p, te_y, a.batch)

    model = (build_improved_model if a.arch == "improved" else build_baseline_model)()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"{a.arch}_cnn.keras")
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    if a.arch == "improved":
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True))
    hist = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cbs, shuffle=False)
    best = keras.models.load_model(ckpt)
    loss, acc = best.evaluate(test, verbose=0)
    print(f"\nTest accuracy of saved model: {acc:.4f}")
    json.dump({"history": hist.history, "test_accuracy": float(acc), "classes": CATEGORIES},
              open(os.path.join(a.out, f"{a.arch}_history.json"), "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
