"""Prediction uncertainty and calibration (NumPy; works from probabilities only)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize_scalar


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def predictive_stats(probs: np.ndarray) -> dict:
    """Top-1, top-2, margin and normalised entropy for ONE probability vector."""
    order = np.argsort(-probs)
    k = len(probs)
    return {"top1": int(order[0]), "top2": int(order[1]),
            "p1": float(probs[order[0]]), "p2": float(probs[order[1]]),
            "margin": float(probs[order[0]] - probs[order[1]]),
            "entropy_norm": float(entropy(probs) / np.log(k))}


# ---------------------------------------------------------------------------------------
# Test-time augmentation (works for any model, including ones without dropout)
# ---------------------------------------------------------------------------------------
def augment_once(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-6, 6, size=2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    out = out * rng.uniform(0.92, 1.08)
    return np.clip(out, 0, 1).astype(np.float32)


def tta_uncertainty(img: np.ndarray, predict_fn, base_class: int, n: int = 16, seed: int = 0) -> dict:
    """Stability of the prediction under small, plausible image changes."""
    rng = np.random.default_rng(seed)
    batch = np.stack([augment_once(img, rng) for _ in range(n)])
    probs = np.asarray(predict_fn(batch))
    return sample_uncertainty(probs, base_class)


def sample_uncertainty(prob_samples: np.ndarray, base_class: int) -> dict:
    """Summaries of a stack of probability vectors (from TTA or MC-dropout)."""
    mean_p = prob_samples.mean(axis=0)
    total = entropy(mean_p)
    aleatoric = entropy(prob_samples, axis=-1).mean()
    return {"mean_probs": mean_p.tolist(),
            "agreement": float(np.mean(prob_samples.argmax(axis=1) == base_class)),
            "std_top": float(prob_samples[:, base_class].std()),
            "mutual_information": float(max(total - aleatoric, 0.0)),   # epistemic part
            "n": int(len(prob_samples))}


# ---------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            acc, c = correct[sel].mean(), conf[sel].mean()
            ece += sel.mean() * abs(acc - c)
            rows.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "accuracy": float(acc), "confidence": float(c)})
    return {"ece": float(ece), "bins": rows, "accuracy": float(correct.mean()), "mean_confidence": float(conf.mean())}


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """softmax(log p / T): identical to rescaling the logits by 1/T."""
    z = np.log(np.clip(probs, 1e-12, 1.0)) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): minimise validation NLL over one scalar."""
    labels = np.asarray(labels).astype(int)

    def nll(T):
        q = apply_temperature(probs, T)
        return -np.mean(np.log(np.clip(q[np.arange(len(labels)), labels], 1e-12, 1)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)
