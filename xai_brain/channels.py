"""Internal-feature evidence: WHICH learned feature channels carry the decision?

For the last convolution (128 / 192 channels) we rank channels by
``Grad-CAM weight x mean activation`` (their signed contribution to the class score), then
verify causally: switch the top-k channels off and measure the fall in P(class).
"""
from __future__ import annotations

import numpy as np

from .saliency import GradientExplainers
from .utils import minmax, resize_map


def _ablate(layer, base_weights, channels):
    """Set the layer to ``base_weights`` with the given output channels zeroed
    (kernel and bias -> the ReLU output of those channels is exactly 0)."""
    new = [w.copy() for w in base_weights]
    new[0][..., channels] = 0.0
    if len(new) > 1:
        new[1][channels] = 0.0
    layer.set_weights(new)


def channel_evidence(ge: GradientExplainers, img: np.ndarray, class_idx: int, predict_fn,
                     top_k: int = 5) -> dict:
    A, g = ge._conv_and_grads(img, class_idx)                  # (h, w, K)
    alpha = g.mean(axis=(0, 1))
    contribution = alpha * A.mean(axis=(0, 1))
    order = np.argsort(-contribution)
    top = [int(k) for k in order[:top_k]]
    maps = [minmax(resize_map(A[..., k], ge.size)) for k in top]

    layer = ge.conv
    activation = getattr(getattr(layer, "activation", None), "__name__", "")
    p0 = float(predict_fn(img[None])[0, class_idx])
    result = {"channels": top, "contribution": [float(contribution[k]) for k in top],
              "mean_activation": [float(A[..., k].mean()) for k in top], "p_orig": p0,
              "ablation_supported": activation == "relu"}
    arrays = {"maps": maps}
    if result["ablation_supported"]:
        saved = [w.copy() for w in layer.get_weights()]
        try:
            single = []
            for k in top:
                _ablate(layer, saved, [k])
                single.append(float(predict_fn(img[None])[0, class_idx]))
            _ablate(layer, saved, top)
            p_all = float(predict_fn(img[None])[0, class_idx])
            # control: the same number of RANDOM channels -> is the drop specific to the top ones?
            rng = np.random.default_rng(0)
            ctrl = []
            for _ in range(10):
                _ablate(layer, saved, rng.choice(A.shape[-1], size=top_k, replace=False).tolist())
                ctrl.append(p0 - float(predict_fn(img[None])[0, class_idx]))
        finally:
            layer.set_weights(saved)
        result.update({"p_without_each": single, "drop_each": [p0 - p for p in single],
                       "p_without_top_k": p_all, "drop_top_k": p0 - p_all,
                       "random_channel_drop_mean": float(np.mean(ctrl))})
    result["arrays"] = arrays
    return result
