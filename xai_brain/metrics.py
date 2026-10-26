"""Quantitative checks of an explanation: faithfulness, agreement, stability."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .evidence import make_filler
from .utils import resize_map


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def deletion_insertion(img: np.ndarray, saliency: np.ndarray, predict_fn, class_idx: int,
                       steps: int = 20, fill_mode: str = "mean") -> dict:
    """Deletion / insertion curves (Petsiuk et al., RISE).

    Deletion  : remove the most important pixels first; a faithful map makes P(class)
                collapse quickly  -> LOW area under the curve is good.
    Insertion : start from a blurred image and reveal the most important pixels first;
                a faithful map makes P(class) rise quickly -> HIGH area is good.
    ``faithfulness = AUC_insertion - AUC_deletion`` (higher = better).
    """
    sal = resize_map(saliency, img.shape[:2]) if saliency.shape != img.shape[:2] else saliency
    order = np.argsort(-sal.ravel(), kind="stable")           # most important first
    n = order.size
    filler = make_filler(img, "blur" if fill_mode == "inpaint" else fill_mode)   # brain mask computed inside for "mean"
    fracs = np.linspace(0, 1, steps + 1)
    flat_img, flat_fill = img.reshape(n, -1), filler.reshape(n, -1)
    dels, ins = [], []
    for f in fracs:
        k = int(round(f * n))
        d = flat_img.copy()
        d[order[:k]] = flat_fill[order[:k]]
        dels.append(d.reshape(img.shape))
        i = flat_fill.copy()
        i[order[:k]] = flat_img[order[:k]]
        ins.append(i.reshape(img.shape))
    probs = np.asarray(predict_fn(np.stack(dels + ins).astype(np.float32)))[:, class_idx]
    d_curve, i_curve = probs[: len(fracs)], probs[len(fracs):]
    a_del, a_ins = _auc(d_curve, fracs), _auc(i_curve, fracs)
    return {"fractions": fracs.tolist(), "deletion": d_curve.tolist(), "insertion": i_curve.tolist(),
            "auc_deletion": a_del, "auc_insertion": a_ins, "faithfulness": a_ins - a_del}


def map_correlation(a: np.ndarray, b: np.ndarray, size: int = 56) -> float:
    """Spearman rank correlation of two maps (down-sampled for speed)."""
    a_s, b_s = resize_map(a, (size, size)).ravel(), resize_map(b, (size, size)).ravel()
    if a_s.std() < 1e-8 or b_s.std() < 1e-8:
        return 0.0
    rho = spearmanr(a_s, b_s).statistic
    return float(0.0 if np.isnan(rho) else rho)


def top_iou(a: np.ndarray, b: np.ndarray, frac: float = 0.2) -> float:
    ta = a >= np.quantile(a, 1 - frac)
    tb = b >= np.quantile(b, 1 - frac)
    union = (ta | tb).sum()
    return float((ta & tb).sum() / union) if union else 0.0


def agreement(maps: dict) -> dict:
    """Pairwise Spearman / top-20 % IoU between explanation methods."""
    names = list(maps)
    k = len(names)
    rho, iou = np.eye(k), np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rho[i, j] = rho[j, i] = map_correlation(maps[names[i]], maps[names[j]])
            iou[i, j] = iou[j, i] = top_iou(maps[names[i]], maps[names[j]])
    off = ~np.eye(k, dtype=bool)
    return {"names": names, "spearman": rho, "iou": iou,
            "mean_spearman": float(rho[off].mean()) if k > 1 else float("nan"),
            "mean_iou": float(iou[off].mean()) if k > 1 else float("nan")}


def stability(explain_fn, img: np.ndarray, class_idx: int, base_map: np.ndarray | None = None,
              n: int = 5, sigma: float = 0.02, seed: int = 0) -> dict:
    """Does the explanation survive imperceptible noise? (mean Spearman vs. clean map)."""
    rng = np.random.default_rng(seed)
    base = explain_fn(img, class_idx) if base_map is None else base_map
    rhos = []
    for _ in range(n):
        noisy = np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)
        rhos.append(map_correlation(base, explain_fn(noisy, class_idx)))
    return {"mean": float(np.mean(rhos)), "min": float(np.min(rhos)), "values": rhos}
