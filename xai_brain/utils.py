"""Small NumPy/OpenCV helpers shared by every module (no TensorFlow here)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata


def to_float01(img) -> np.ndarray:
    """Return an HxWx3 float32 image in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def gray(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=-1)


def minmax(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < eps:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def resize_map(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D map to ``size=(H, W)``."""
    return cv2.resize(np.asarray(m, dtype=np.float32), (size[1], size[0]),
                      interpolation=cv2.INTER_LINEAR)


def rank_normalize(m: np.ndarray) -> np.ndarray:
    """Replace values by their rank in [0, 1] - makes different methods comparable."""
    flat = rankdata(m.ravel(), method="average")
    flat = (flat - 1) / max(len(flat) - 1, 1)
    return flat.reshape(m.shape).astype(np.float32)


def brain_mask(img: np.ndarray, thr: float = 0.08) -> tuple[np.ndarray, bool]:
    """Rough brain/skull foreground mask. Returns (mask, ok).

    ``ok`` is False when the heuristic failed and the whole image was returned.
    """
    g = cv2.GaussianBlur(gray(img), (5, 5), 0)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.median(border) > thr:                       # background is not black -> Otsu
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(g))
    m = g > thr
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=3)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    full = np.ones_like(m, dtype=bool)
    if n == 0:
        return full, False
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    frac = m.mean()
    if frac < 0.05 or frac > 0.97:
        return full, False
    return m, True


def hotspot_mask(saliency: np.ndarray, top_frac: float = 0.10, min_area: int = 30,
                 max_components: int = 3) -> np.ndarray:
    """Binary mask of the strongest evidence: top ``top_frac`` of pixels, cleaned."""
    s = np.asarray(saliency, dtype=np.float32)
    thr = np.quantile(s, 1.0 - top_frac)
    m = s >= thr
    if thr <= s.min():                                 # flat map
        return np.zeros_like(m)
    m = ndi.binary_opening(m, structure=np.ones((3, 3)))
    lab, n = ndi.label(m)
    if n == 0:
        m = s >= thr
        lab, n = ndi.label(m)
        if n == 0:
            return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = [i + 1 for i in np.argsort(-sizes)[:max_components] if sizes[i] >= min_area]
    if not keep:
        keep = [1 + int(np.argmax(sizes))]
    return np.isin(lab, keep)


def overlay(img: np.ndarray, m: np.ndarray, cmap: str = "jet", alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] map over an image (returns float RGB)."""
    import matplotlib
    colored = matplotlib.colormaps[cmap](np.clip(m, 0, 1))[..., :3]
    return np.clip((1 - alpha) * img + alpha * colored, 0, 1)
