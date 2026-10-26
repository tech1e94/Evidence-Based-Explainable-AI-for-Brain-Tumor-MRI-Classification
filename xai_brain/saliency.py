"""Attribution / saliency methods.

Model-agnostic (NumPy only, need just ``predict_fn``):
    occlusion_map, lime_map
TensorFlow / Keras (``GradientExplainers``):
    Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, SmoothGrad, attention-gate map,
    plus ``parameter_randomization_test`` (sanity check of the explanation method itself).

Every explainer returned by ``build_explainers`` has the signature
``fn(img[H,W,3] float32 in [0,1], class_idx) -> map[H,W] float32 in [0,1]``
(1 = most evidence FOR ``class_idx``).
"""
from __future__ import annotations

import cv2
import numpy as np

from .metrics import map_correlation
from .utils import minmax, resize_map

METHOD_LABELS = {
    "gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "scorecam": "Score-CAM",
    "ig": "Integrated Gradients", "smoothgrad": "SmoothGrad", "occlusion": "Occlusion",
    "lime": "LIME", "attention": "Attention gate",
}


# ======================================================================================
# Model-agnostic methods
# ======================================================================================
def occlusion_map(img: np.ndarray, predict_fn, class_idx: int, patch: int = 32, stride: int = 16,
                  chunk: int = 64) -> np.ndarray:
    """Slide a grey patch over the image; the drop in P(class) is the evidence in that patch."""
    h, w = img.shape[:2]
    fill = img.mean(axis=(0, 1))
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)
    base = float(np.asarray(predict_fn(img[None]))[0, class_idx])
    coords = [(y, x) for y in ys for x in xs]
    heat = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for i in range(0, len(coords), chunk):
        part = coords[i:i + chunk]
        batch = np.repeat(img[None], len(part), axis=0)
        for j, (y, x) in enumerate(part):
            batch[j, y:y + patch, x:x + patch] = fill
        p = np.asarray(predict_fn(batch))[:, class_idx]
        for (y, x), pj in zip(part, p):
            heat[y:y + patch, x:x + patch] += base - pj
            cnt[y:y + patch, x:x + patch] += 1
    heat = np.clip(heat / np.maximum(cnt, 1), 0, None)
    heat = cv2.GaussianBlur(heat, (0, 0), 4)
    return minmax(heat)


def lime_map(img: np.ndarray, predict_fn, class_idx: int, num_samples: int = 800, seed: int = 0) -> np.ndarray:
    """LIME super-pixel weights painted back onto the image (positive weights only)."""
    from lime import lime_image                                # lazy: optional dependency
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    kwargs = dict(labels=(class_idx,), top_labels=None, hide_color=0, num_samples=num_samples,
                  batch_size=64, random_seed=seed)
    try:
        exp = explainer.explain_instance(img.astype("double"), predict_fn, progress_bar=False, **kwargs)
    except TypeError:                                          # very old lime versions
        exp = explainer.explain_instance(img.astype("double"), predict_fn, **kwargs)
    m = np.zeros(img.shape[:2], np.float32)
    for seg, wgt in exp.local_exp[class_idx]:
        m[exp.segments == seg] = max(float(wgt), 0.0)
    return minmax(cv2.GaussianBlur(m, (0, 0), 2))


# ======================================================================================
# Layer discovery (works on the original notebook's model AND on the improved one)
# ======================================================================================
def _act_name(layer) -> str:
    act = getattr(layer, "activation", None)
    return getattr(act, "__name__", str(act))


def find_attention_gate(model):
    """The 1-filter sigmoid Conv2D (the spatial attention gate), or None."""
    from tensorflow import keras
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) and layer.filters == 1 and _act_name(layer) == "sigmoid":
            return layer
    return None


def find_conv_layer(model, name: str | None = None):
    """Last 'real' convolution (skips the 1-filter attention gate)."""
    from tensorflow import keras
    if name:
        return model.get_layer(name)
    try:
        return model.get_layer("last_conv")
    except ValueError:
        pass
    gate = find_attention_gate(model)
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D) and layer is not gate:
            return layer
    raise ValueError("No Conv2D layer found - pass conv_layer explicitly.")


# ======================================================================================
# TensorFlow explainers
# ======================================================================================
class GradientExplainers:
    def __init__(self, model, predict_fn=None, conv_layer: str | None = None):
        import tensorflow as tf
        from tensorflow import keras
        self.tf, self.keras, self.model = tf, keras, model
        self.predict_fn = predict_fn
        self.conv = find_conv_layer(model, conv_layer)
        last = model.layers[-1]
        self._manual_logits = isinstance(last, keras.layers.Dense) and _act_name(last) == "softmax"
        outs = [self.conv.output, last.input if self._manual_logits else model.output, model.output]
        self.sub = keras.Model(model.input, outs)
        if self._manual_logits:
            W, b = last.get_weights()
            self.W, self.b = tf.constant(W, tf.float32), tf.constant(b, tf.float32)
        gate = find_attention_gate(model)
        self.gate_model = keras.Model(model.input, gate.output) if gate is not None else None
        self.size = tuple(model.input_shape[1:3])

    # --- forward ---------------------------------------------------------------------
    def _forward(self, x):
        conv_out, h, out = self.sub(x, training=False)
        logits = self.tf.matmul(h, self.W) + self.b if self._manual_logits else out
        return conv_out, logits

    def _conv_and_grads(self, img, class_idx):
        tf = self.tf
        x = tf.convert_to_tensor(img[None], tf.float32)
        with tf.GradientTape() as tape:
            conv_out, logits = self._forward(x)
            score = logits[:, class_idx]
        grads = tape.gradient(score, conv_out)
        return conv_out[0].numpy(), grads[0].numpy()

    def _upsample(self, cam):
        return minmax(resize_map(np.maximum(cam, 0), self.size))

    # --- CAM family ---------------------------------------------------------------------
    def gradcam(self, img, class_idx):
        A, g = self._conv_and_grads(img, class_idx)
        w = g.mean(axis=(0, 1))
        return self._upsample((A * w).sum(-1))

    def gradcampp(self, img, class_idx):
        A, g = self._conv_and_grads(img, class_idx)
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + A.sum(axis=(0, 1), keepdims=True) * g3
        denom = np.where(denom != 0, denom, 1.0)
        w = ((g2 / denom) * np.maximum(g, 0)).sum(axis=(0, 1))
        return self._upsample((A * w).sum(-1))

    def scorecam(self, img, class_idx, max_channels: int = 48):
        """Gradient-free: weight each channel by how much keeping only it preserves P(class)."""
        if self.predict_fn is None:
            raise ValueError("scorecam needs predict_fn")
        A, _ = self._conv_and_grads(img, class_idx)
        keep = np.argsort(-A.reshape(-1, A.shape[-1]).std(axis=0))[:max_channels]
        maps = np.stack([resize_map(A[..., k], self.size) for k in keep])          # (n,H,W)
        norm = np.stack([minmax(m) for m in maps])
        masked = (img[None] * norm[..., None]).astype(np.float32)
        base = float(np.asarray(self.predict_fn(np.zeros_like(img)[None]))[0, class_idx])
        scores = np.asarray(self.predict_fn(masked))[:, class_idx] - base
        w = np.exp(scores - scores.max())
        w /= w.sum()
        return minmax(np.maximum((w[:, None, None] * maps).sum(0), 0))

    # --- input-gradient family --------------------------------------------------------------
    def _input_grads(self, batch, class_idx, chunk: int = 16):
        tf, out = self.tf, []
        for i in range(0, len(batch), chunk):
            x = tf.convert_to_tensor(batch[i:i + chunk], tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x)
                _, logits = self._forward(x)
                score = logits[:, class_idx]
            out.append(tape.gradient(score, x).numpy())
        return np.concatenate(out)

    def integrated_gradients(self, img, class_idx, steps: int = 32, smooth: float = 3.0):
        base = np.zeros_like(img)
        alphas = (np.arange(steps) + 0.5) / steps
        path = (base[None] + alphas[:, None, None, None] * (img - base)[None]).astype(np.float32)
        avg = self._input_grads(path, class_idx).mean(axis=0)
        attr = ((img - base) * avg).sum(-1)                       # signed, summed over channels
        attr = cv2.GaussianBlur(np.maximum(attr, 0).astype(np.float32), (0, 0), smooth)
        return minmax(attr)

    def smoothgrad(self, img, class_idx, n: int = 16, sigma: float = 0.05, smooth: float = 2.0):
        rng = np.random.default_rng(0)
        noisy = np.clip(img[None] + rng.normal(0, sigma, (n,) + img.shape), 0, 1).astype(np.float32)
        g = np.abs(self._input_grads(noisy, class_idx)).mean(axis=0).max(-1)
        return minmax(cv2.GaussianBlur(g.astype(np.float32), (0, 0), smooth))

    # --- built-in attention ----------------------------------------------------------------------
    def attention(self, img, class_idx=None):
        if self.gate_model is None:
            raise ValueError("model has no sigmoid attention gate")
        a = self.gate_model(self.tf.convert_to_tensor(img[None], self.tf.float32), training=False)
        return minmax(resize_map(a.numpy()[0, ..., 0], self.size))


def build_explainers(model, predict_fn, methods=("gradcam", "gradcampp", "scorecam", "ig", "occlusion")):
    """Dictionary ``name -> fn(img, class_idx) -> map`` for the requested methods."""
    ge = GradientExplainers(model, predict_fn)
    table = {
        "gradcam": ge.gradcam, "gradcampp": ge.gradcampp, "scorecam": ge.scorecam,
        "ig": ge.integrated_gradients, "smoothgrad": ge.smoothgrad,
        "occlusion": lambda im, c: occlusion_map(im, predict_fn, c),
        "lime": lambda im, c: lime_map(im, predict_fn, c),
    }
    unknown = set(methods) - set(table)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(table)}")
    return {m: table[m] for m in methods}, ge


# ======================================================================================
# Sanity check of the explanation method itself (Adebayo et al., 2018)
# ======================================================================================
def parameter_randomization_test(model, images, class_indices, seed: int = 0) -> dict:
    """A valid explanation must depend on the learned weights.

    Grad-CAM maps of the trained model are compared with maps from (a) a randomly
    initialised copy and (b) the trained model whose final layer is re-randomised.
    ``mean_abs_rho`` close to 0 -> maps change a lot -> the explanation reflects what
    the model learned (PASS). Close to 1 -> the map is mostly an image edge detector.
    """
    from tensorflow import keras
    rng = np.random.default_rng(seed)
    ge = GradientExplainers(model)
    rand_model = keras.models.clone_model(model)                       # fresh random weights
    head_model = keras.models.clone_model(model)
    head_model.set_weights(model.get_weights())
    last = head_model.layers[-1]
    W, b = last.get_weights()
    last.set_weights([rng.normal(0, 1 / np.sqrt(W.shape[0]), W.shape).astype(W.dtype), np.zeros_like(b)])
    ge_rand, ge_head = GradientExplainers(rand_model), GradientExplainers(head_model)
    rho_full, rho_head = [], []
    for img, c in zip(images, class_indices):
        ref = ge.gradcam(img, int(c))
        rho_full.append(map_correlation(ref, ge_rand.gradcam(img, int(c))))
        rho_head.append(map_correlation(ref, ge_head.gradcam(img, int(c))))
    res = {"mean_abs_rho_full_random": float(np.mean(np.abs(rho_full))),
           "mean_abs_rho_head_random": float(np.mean(np.abs(rho_head))),
           "rho_full": rho_full, "rho_head": rho_head}
    res["passed"] = bool(res["mean_abs_rho_full_random"] < 0.5)
    return res
