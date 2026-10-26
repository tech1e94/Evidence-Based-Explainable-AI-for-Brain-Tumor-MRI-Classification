"""Model definitions and thin TensorFlow helpers (the only module that builds networks)."""
from __future__ import annotations

import numpy as np

from .config import CATEGORIES, IMG_SIZE


def _attention_block(x, name_prefix="attention"):
    from tensorflow.keras import layers
    gate = layers.Conv2D(1, (1, 1), activation="sigmoid", name=f"{name_prefix}_gate")(x)
    return layers.Multiply(name=f"{name_prefix}_out")([x, gate])


def build_baseline_model(num_classes: int = len(CATEGORIES), img_size: int = IMG_SIZE):
    """Exactly the architecture of the original notebook, with named layers.

    (Layer names are added so the attention gate and last conv can be found by name.)
    """
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", name="last_conv")(x)
    x = _attention_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu", name="embedding")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="baseline_cnn_attention")


def build_improved_model(num_classes: int = len(CATEGORIES), img_size: int = IMG_SIZE, dropout: float = 0.3):
    """CNN + BatchNorm + attention gate + dropout (dropout also enables MC-dropout).

    A stride-2 stem keeps it ~3x cheaper than a full-resolution first block (CPU-friendly); the
    final feature map is 28x28 for a 224 input. ``last_conv`` has a plain ReLU (no BatchNorm after
    it) so channel-ablation works. BatchNorm momentum 0.9 lets the running statistics settle
    within the first epoch (the default 0.99 gives poor validation scores early in training).
    """
    from tensorflow.keras import layers, models

    def conv_bn(x, filters, strides=1):
        x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.Activation("relu")(x)

    def block(x, filters, pool=True):
        x = conv_bn(conv_bn(x, filters), filters)
        return layers.MaxPooling2D()(x) if pool else x

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = conv_bn(inputs, 32, strides=2)              # 224 -> 112
    x = block(x, 32)                                # -> 56
    x = block(x, 64)                                # -> 28
    x = block(x, 128, pool=False)                   # 28
    x = layers.Conv2D(192, 3, padding="same", activation="relu", name="last_conv")(x)
    x = _attention_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="improved_cnn_attention")


def load_model(path: str):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def make_predict_fn(model, batch_size: int = 64):
    """``fn(images[N,H,W,3] in [0,1]) -> probabilities[N,K]`` (deterministic, dropout off)."""
    def predict(x):
        x = np.asarray(x, dtype=np.float32)
        out = [np.asarray(model(x[i:i + batch_size], training=False))
               for i in range(0, len(x), batch_size)]
        return np.concatenate(out, axis=0)
    return predict


def make_mc_dropout_fn(model, n_samples: int = 20, batch_size: int = 32):
    """Monte-Carlo dropout: ``fn(img) -> samples[n_samples, K]`` (needs Dropout layers).

    A clone of the model is built whose Dropout layers are always active while every other
    layer (notably BatchNormalization) stays in inference mode - calling the model with
    ``training=True`` would also switch BatchNorm to batch statistics and corrupt the samples.
    """
    from tensorflow import keras

    class MCDropout(keras.layers.Dropout):
        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

    def clone_fn(layer):
        cls = MCDropout if isinstance(layer, keras.layers.Dropout) else layer.__class__
        return cls.from_config(layer.get_config())

    mc_model = keras.models.clone_model(model, clone_function=clone_fn)
    mc_model.set_weights(model.get_weights())

    def sample(img):
        x = np.repeat(np.asarray(img, dtype=np.float32)[None], n_samples, axis=0)
        return np.concatenate([np.asarray(mc_model(x[i:i + batch_size], training=False))
                               for i in range(0, n_samples, batch_size)], axis=0)
    return sample


def has_dropout(model) -> bool:
    from tensorflow import keras
    return any(isinstance(layer, keras.layers.Dropout) for layer in model.layers)


def embedding_model(model):
    """Sub-model returning the penultimate (pre-classifier) feature vector."""
    from tensorflow import keras
    for layer in reversed(model.layers[:-1]):
        if isinstance(layer, (keras.layers.Dense, keras.layers.GlobalAveragePooling2D)):
            return keras.Model(model.input, layer.output)
    raise ValueError("no penultimate Dense/GAP layer found")


def get_embeddings(model, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    emb = embedding_model(model)
    return np.concatenate([np.asarray(emb(images[i:i + batch_size], training=False))
                           for i in range(0, len(images), batch_size)], axis=0)
