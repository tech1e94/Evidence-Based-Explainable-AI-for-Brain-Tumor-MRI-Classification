"""Dataset helpers: robust folder lookup, deterministic train/validation split, loading.

``find_split_dir`` is case-insensitive, so ``archive/Testing`` and ``archive/testing``
both work (the original notebook used both spellings, which breaks on Linux/Colab).
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .config import CATEGORIES, IMG_SIZE
from .utils import to_float01

_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_split_dir(base: str, name: str) -> str:
    """Return ``base/<name>`` matching the folder name case-insensitively."""
    for entry in os.listdir(base):
        if entry.lower() == name.lower() and os.path.isdir(os.path.join(base, entry)):
            return os.path.join(base, entry)
    raise FileNotFoundError(f"No folder called '{name}' (any case) inside '{base}'. "
                            f"Found: {os.listdir(base)}")


def list_images(split_dir: str, categories: Sequence[str] = CATEGORIES):
    """Return ``(paths, labels)`` sorted per class."""
    paths, labels = [], []
    for idx, cat in enumerate(categories):
        folder = None
        for entry in os.listdir(split_dir):
            if entry.lower() == cat.lower():
                folder = os.path.join(split_dir, entry)
        if folder is None:
            raise FileNotFoundError(f"Class folder '{cat}' not found in {split_dir}")
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(_EXT):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    return paths, np.array(labels, dtype=np.int64)


def train_val_split(paths, labels, val_frac: float = 0.1):
    """Per class, the LAST ``val_frac`` of the sorted files become validation.

    This mirrors Keras' ``ImageDataGenerator(validation_split=0.1)`` used in the
    original notebook, so the validation images are (approximately) the same ones.
    """
    paths = np.array(paths)
    tr_idx, va_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n_val = int(round(len(idx) * val_frac))
        tr_idx.extend(idx[: len(idx) - n_val])
        va_idx.extend(idx[len(idx) - n_val:])
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
    return (list(paths[tr_idx]), labels[tr_idx]), (list(paths[va_idx]), labels[va_idx])


def load_image(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """Load one image as RGB float32 in [0, 1] (RGB, matching how the model was trained)."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((size, size))
    return to_float01(np.asarray(im))


def load_images(paths: Sequence[str], size: int = IMG_SIZE) -> np.ndarray:
    return np.stack([load_image(p, size) for p in paths]).astype(np.float32)


def sample_per_class(paths, labels, n_per_class: int, seed: int = 0):
    """Random, class-balanced subset (for the dataset-level XAI evaluation)."""
    rng = np.random.default_rng(seed)
    paths = np.array(paths)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep.extend(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    keep = np.array(sorted(keep))
    return list(paths[keep]), labels[keep]


def make_tf_dataset(paths, labels, batch_size=32, size=IMG_SIZE, augment=False, shuffle=False, seed=0):
    """tf.data pipeline (imports TensorFlow lazily). Pixel values are scaled to [0, 1]."""
    import tensorflow as tf

    paths = tf.constant(list(paths))
    labels = tf.constant(np.asarray(labels, dtype=np.int32))
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)

    def _load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        return tf.cast(img, tf.float32) / 255.0, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.04, seed=seed),
            tf.keras.layers.RandomZoom(0.08, seed=seed),
            tf.keras.layers.RandomContrast(0.10, seed=seed),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
