"""Turn a saliency map into *measurable evidence*.

Two kinds of evidence are produced:

1. ``region_features``   - WHAT is in the region the model relied on (location, size,
   brightness relative to the rest of the brain, texture, shape, boundary sharpness,
   left/right asymmetry, share of evidence outside the brain / on the skull rim).
2. ``counterfactual_tests`` - does the model REALLY depend on that region?
   Remove it (necessity), keep only it (sufficiency) and compare against random
   same-shape regions (specificity). This is causal evidence, not just a heat-map.

Everything is NumPy/SciPy/scikit-image, driven by ``predict_fn(images)->probabilities``.

Important caveat (also repeated in the report): the region is *where the model looked*,
it is not a radiologist's tumour segmentation. Features are measured on an
intensity-refined version of that region and should be read as descriptive evidence.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

from .utils import gray

# Features used by the surrogate model / class profiles (all numeric).
FEATURE_NAMES = [
    "area_frac", "dist_center", "depth", "midline_offset", "rel_y",
    "contrast_z", "heterogeneity", "texture_entropy", "glcm_contrast", "glcm_homogeneity",
    "solidity", "eccentricity", "compactness", "boundary_sharpness", "asymmetry_ratio",
    "n_components", "in_brain_mass", "rim_mass",
]

FEATURE_DESCRIPTIONS = {
    "area_frac": "size of the evidence region (fraction of brain area)",
    "dist_center": "distance of the region from the brain centre (0 = centre, 1 = edge)",
    "depth": "depth inside the brain (0 = at the surface, 1 = deepest point)",
    "midline_offset": "horizontal distance from the brain midline",
    "rel_y": "vertical position (-1 = top of the brain, +1 = bottom)",
    "contrast_z": "brightness vs. the surrounding brain (in standard deviations)",
    "heterogeneity": "intensity variation inside the region (std / mean)",
    "texture_entropy": "texture entropy (disorder) of the region",
    "glcm_contrast": "local texture contrast (GLCM)",
    "glcm_homogeneity": "texture homogeneity (GLCM)",
    "solidity": "shape solidity (1 = convex, solid blob)",
    "eccentricity": "shape eccentricity (0 = round, 1 = elongated)",
    "compactness": "shape compactness (1 = perfect circle, low = irregular)",
    "boundary_sharpness": "edge strength along the region boundary vs. rest of brain",
    "asymmetry_ratio": "left/right asymmetry inside region vs. whole brain",
    "n_components": "number of separate evidence blobs",
    "in_brain_mass": "share of the explanation lying inside the brain",
    "rim_mass": "share of the explanation on the outer brain rim / skull edge",
}


# --------------------------------------------------------------------------------------
# Image editing helpers (used for removal / retention tests)
# --------------------------------------------------------------------------------------
def make_filler(img: np.ndarray, mode: str = "mean", brain: np.ndarray | None = None) -> np.ndarray:
    """What to paint over removed regions.

    'mean' (default): median brain-tissue intensity inside the brain, black outside. A plain blur
        is NOT used by default because a broad bright lesion largely survives blurring, so
        "removing" it would not remove the evidence.
    'blur': heavily blurred copy of the image (classic RISE baseline).
    """
    if mode == "mean":
        if brain is None:
            from .utils import brain_mask
            brain, _ = brain_mask(img)
        tissue = np.median(img[brain], axis=0) if brain.any() else img.mean(axis=(0, 1))
        out = np.zeros_like(img)
        out[brain] = tissue
        return out.astype(np.float32)
    if mode in ("blur", "inpaint"):                      # inpaint is handled in remove_region
        return cv2.GaussianBlur(img, (0, 0), sigmaX=12)
    raise ValueError(f"unknown fill mode {mode}")


def remove_region(img: np.ndarray, mask: np.ndarray, filler: np.ndarray, mode: str = "blur",
                  feather: float = 2.0) -> np.ndarray:
    """Replace ``mask`` pixels of ``img`` with the filler (soft edge)."""
    mask = mask.astype(bool)
    if not mask.any():
        return img.copy()
    if mode == "inpaint":
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        dil = ndi.binary_dilation(mask, iterations=2).astype(np.uint8) * 255
        out = cv2.inpaint(u8, dil, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        return out
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)[..., None]
    alpha = np.clip(alpha, 0, 1)
    return (img * (1 - alpha) + filler * alpha).astype(np.float32)


def keep_only_region(img: np.ndarray, mask: np.ndarray, filler: np.ndarray, feather: float = 2.0) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)[..., None]
    alpha = np.clip(alpha, 0, 1)
    return (img * alpha + filler * (1 - alpha)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Counterfactual evidence
# --------------------------------------------------------------------------------------
def random_control_masks(hotspot: np.ndarray, brain: np.ndarray, n: int, rng: np.random.Generator,
                         max_overlap: float = 0.2, min_inside: float = 0.6, tries: int = 500):
    """Translate the hotspot shape to random places (same size and shape, mostly inside brain)."""
    ys, xs = np.nonzero(hotspot)
    h, w = hotspot.shape
    masks = []
    for _ in range(tries):
        dy = int(rng.integers(-h // 2, h // 2))
        dx = int(rng.integers(-w // 2, w // 2))
        ny, nx = ys + dy, xs + dx
        if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
            continue
        m = np.zeros_like(hotspot, dtype=bool)
        m[ny, nx] = True
        if (m & hotspot).sum() / hotspot.sum() > max_overlap:
            continue
        if (m & brain).sum() / m.sum() < min_inside:
            continue
        masks.append(m)
        if len(masks) == n:
            break
    return masks


def counterfactual_tests(img: np.ndarray, hotspot: np.ndarray, brain: np.ndarray, predict_fn,
                         class_idx: int, n_random: int = 30, fill_mode: str = "mean",
                         seed: int = 0) -> dict:
    """Necessity / sufficiency / specificity of the hotspot for ``class_idx``.

    * necessity   : P(class) after the hotspot is removed -> ``drop`` = p_orig - p_removed
    * sufficiency : P(class) when ONLY the hotspot is kept -> ``sufficiency`` = p_kept / p_orig
    * specificity : the same removal on ``n_random`` random regions of identical shape;
                    ``p_value`` = (1 + #{random drops >= hotspot drop}) / (1 + n_random)
    """
    rng = np.random.default_rng(seed)
    filler = make_filler(img, "blur" if fill_mode == "inpaint" else fill_mode, brain)
    removed = remove_region(img, hotspot, filler, fill_mode)
    kept = keep_only_region(img, hotspot, filler)
    controls = random_control_masks(hotspot, brain, n_random, rng)
    ctrl_imgs = [remove_region(img, m, filler, fill_mode) for m in controls]

    batch = np.stack([img, removed, kept] + ctrl_imgs).astype(np.float32)
    probs = np.asarray(predict_fn(batch))
    p_orig, p_removed, p_kept = probs[0, class_idx], probs[1, class_idx], probs[2, class_idx]
    ctrl_p = probs[3:, class_idx]
    ctrl_drops = p_orig - ctrl_p
    drop = float(p_orig - p_removed)
    n_c = len(ctrl_drops)
    p_value = float((1 + np.sum(ctrl_drops >= drop)) / (1 + n_c)) if n_c else float("nan")
    ctrl_mean = float(ctrl_drops.mean()) if n_c else float("nan")
    return {
        "p_orig": float(p_orig), "p_removed": float(p_removed), "p_kept": float(p_kept),
        "drop": drop,
        "sufficiency": float(p_kept / max(p_orig, 1e-8)),
        "class_after_removal": int(np.argmax(probs[1])),
        "probs_removed": probs[1].tolist(), "probs_kept": probs[2].tolist(),
        "control_drops": ctrl_drops.tolist(), "n_controls": int(n_c),
        "control_mean_drop": ctrl_mean,
        "effect_ratio": float(drop / max(ctrl_mean, 1e-3)) if n_c else float("nan"),
        "p_value": p_value,
        "arrays": {"removed": removed, "kept": kept},
    }


# --------------------------------------------------------------------------------------
# Region features
# --------------------------------------------------------------------------------------
def refine_region(img: np.ndarray, hotspot: np.ndarray, brain: np.ndarray, k: float = 0.75,
                  min_area: int = 40) -> np.ndarray:
    """Inside a slightly dilated hotspot keep pixels clearly different from typical brain tissue.

    A saliency hotspot is blurry (CAMs are low resolution). Intersecting it with an
    intensity criterion gives a crude, tumour-like region so that shape / boundary
    features are not just the shape of a blurred blob.
    """
    g = gray(img)
    inside = g[brain]
    med, sd = float(np.median(inside)), float(inside.std() + 1e-6)
    cand = ndi.binary_dilation(hotspot, iterations=4) & brain & (np.abs(g - med) > k * sd)
    cand = ndi.binary_opening(cand, structure=np.ones((3, 3)))
    lab, n = ndi.label(cand)
    if n:
        sizes = ndi.sum(cand, lab, index=np.arange(1, n + 1))
        cand = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_area])
    return cand if cand.sum() >= min_area else hotspot.copy()


def mass_outside(saliency: np.ndarray, brain: np.ndarray, top_quantile: float = 0.85) -> tuple[float, float]:
    """(share of explanation inside brain, share on the outer rim of the brain).

    Measured on the STRONGEST evidence only (values above ``top_quantile`` of the map, weighted by how
    far above). A raw sum would be dominated by the low-level floor that every saliency map has, which
    makes the number just mirror the brain's share of the image.
    """
    s = np.asarray(saliency, dtype=np.float64)
    s = np.clip(s - np.quantile(s, top_quantile), 0, None)
    tot = s.sum() + 1e-12
    inside = float(s[brain].sum() / tot)
    dt = ndi.distance_transform_edt(brain)
    rim = brain & (dt <= 0.12 * max(dt.max(), 1.0))
    return inside, float(s[rim].sum() / tot)


def region_features(img: np.ndarray, hotspot: np.ndarray, brain: np.ndarray,
                    saliency: np.ndarray | None = None) -> dict:
    """Measure the evidence region. All values are plain floats."""
    g = gray(img)
    h, w = g.shape
    region = refine_region(img, hotspot, brain)
    if region.sum() < 5:
        region = hotspot.copy()
    rest = brain & ~region
    if rest.sum() < 10:
        rest = brain

    ys, xs = np.nonzero(brain)
    cy, cx = ys.mean(), xs.mean()
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    radius = np.sqrt(brain.sum() / np.pi)

    ry, rx = np.nonzero(region)
    ry_c, rx_c = ry.mean(), rx.mean()
    dt = ndi.distance_transform_edt(brain)
    depth = float(dt[int(round(ry_c)), int(round(rx_c))] / max(dt.max(), 1.0))

    mean_in, mean_rest, sd_rest = g[region].mean(), g[rest].mean(), g[rest].std() + 1e-6
    contrast_z = float((mean_in - mean_rest) / sd_rest)

    # texture on the region's bounding box (quantised to 32 grey levels)
    y0, y1, x0, x1 = ry.min(), ry.max() + 1, rx.min(), rx.max() + 1
    crop = (np.clip(g[y0:y1, x0:x1], 0, 1) * 31).astype(np.uint8)
    if crop.shape[0] >= 3 and crop.shape[1] >= 3:
        glcm = graycomatrix(crop, distances=[1], angles=[0, np.pi / 2], levels=32,
                            symmetric=True, normed=True)
        gl_con = float(graycoprops(glcm, "contrast").mean())
        gl_hom = float(graycoprops(glcm, "homogeneity").mean())
    else:
        gl_con, gl_hom = 0.0, 1.0
    hist, _ = np.histogram(g[region], bins=16, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    tex_entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    # shape on the largest connected component
    lab = measure.label(region)
    props = max(measure.regionprops(lab), key=lambda r: r.area)
    perim = max(props.perimeter, 1.0)
    compactness = float(min(4 * np.pi * props.area / perim ** 2, 1.0))

    # boundary sharpness
    sob = np.hypot(ndi.sobel(cv2.GaussianBlur(g, (0, 0), 1.0), axis=0),
                   ndi.sobel(cv2.GaussianBlur(g, (0, 0), 1.0), axis=1))
    boundary = region & ~ndi.binary_erosion(region)
    sharp = float(sob[boundary].mean() / (sob[brain].mean() + 1e-6)) if boundary.any() else 0.0

    # left/right asymmetry about the brain's vertical midline
    shift = int(round(2 * cx - (w - 1)))
    mirrored = np.roll(g[:, ::-1], shift, axis=1)
    mirrored_brain = np.roll(brain[:, ::-1], shift, axis=1)
    valid = brain & mirrored_brain
    asym = np.abs(g - mirrored)
    a_reg = asym[region & valid].mean() if (region & valid).any() else 0.0
    a_all = asym[valid].mean() if valid.any() else 1e-6
    asym_ratio = float(a_reg / (a_all + 1e-6))

    in_mass, rim_mass = (1.0, 0.0)
    if saliency is not None:
        in_mass, rim_mass = mass_outside(saliency, brain)

    n_comp = int(measure.label(hotspot).max())
    feats = {
        "area_frac": float(region.sum() / brain.sum()),
        "dist_center": float(np.hypot(ry_c - cy, rx_c - cx) / radius),
        "depth": depth,
        "midline_offset": float(abs(rx_c - cx) / (bw / 2)),
        "rel_x": float((rx_c - cx) / (bw / 2)),          # signed (kept for text, not a surrogate feature)
        "rel_y": float((ry_c - cy) / (bh / 2)),
        "contrast_z": contrast_z,
        "heterogeneity": float(g[region].std() / (mean_in + 1e-6)),
        "texture_entropy": tex_entropy,
        "glcm_contrast": gl_con,
        "glcm_homogeneity": gl_hom,
        "solidity": float(props.solidity),
        "eccentricity": float(props.eccentricity),
        "compactness": compactness,
        "boundary_sharpness": sharp,
        "asymmetry_ratio": asym_ratio,
        "n_components": float(n_comp),
        "in_brain_mass": in_mass,
        "rim_mass": rim_mass,
    }
    return {"features": feats, "region": region}


# --------------------------------------------------------------------------------------
# Words for numbers
# --------------------------------------------------------------------------------------
def describe_features(f: dict) -> dict:
    """Human-readable phrases for the numeric features (thresholds are descriptive only)."""
    def band(v, cuts, words):
        for c, wd in zip(cuts, words):
            if v < c:
                return wd
        return words[-1]

    horiz = "near the midline" if abs(f["rel_x"]) < 0.25 else (
        "in the left half of the image" if f["rel_x"] < 0 else "in the right half of the image")
    vert = "upper" if f["rel_y"] < -0.25 else ("lower" if f["rel_y"] > 0.25 else "middle")
    depth = band(f["depth"], [0.25, 0.6], ["close to the brain surface", "at mid-depth", "deep inside the brain"])
    size = band(f["area_frac"], [0.02, 0.08, 0.2], ["very small", "small", "medium-sized", "large"])
    if f["contrast_z"] > 1.0:
        inten = f"brighter than the surrounding brain ({f['contrast_z']:+.1f} SD)"
    elif f["contrast_z"] < -1.0:
        inten = f"darker than the surrounding brain ({f['contrast_z']:+.1f} SD)"
    else:
        inten = f"similar in brightness to the surrounding brain ({f['contrast_z']:+.1f} SD)"
    shape = ("round and compact" if f["compactness"] > 0.6 and f["solidity"] > 0.9
             else "elongated" if f["eccentricity"] > 0.9
             else "irregular" if f["solidity"] < 0.8 or f["compactness"] < 0.3 else "moderately regular")
    edge = band(f["boundary_sharpness"], [0.8, 1.3], ["a soft/diffuse boundary", "a moderately defined boundary",
                                                      "a sharp boundary"])
    texture = band(f["heterogeneity"], [0.15, 0.3], ["homogeneous", "mildly heterogeneous", "heterogeneous"])
    asym = ("markedly asymmetric vs. the opposite side" if f["asymmetry_ratio"] > 1.6
            else "roughly symmetric vs. the opposite side" if f["asymmetry_ratio"] < 1.1
            else "mildly asymmetric vs. the opposite side")
    return {"location": f"{vert}, {horiz}, {depth}", "size": size, "intensity": inten,
            "shape": shape, "edge": edge, "texture": texture, "asymmetry": asym}
