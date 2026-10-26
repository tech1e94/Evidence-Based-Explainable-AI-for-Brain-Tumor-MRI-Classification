"""Synthetic MRI-like images + a fake classifier (NumPy only).

Used to unit-test every non-TensorFlow part of the toolkit without data or a GPU.
The fake model really reacts to bright blobs, so removal / occlusion tests are meaningful.
"""
import cv2
import numpy as np

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def make_mri(label: int, rng: np.random.Generator, size: int = 224):
    """Return (img[H,W,3] float32, tumour_mask bool). label 2 = no tumour."""
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2 + rng.normal(0, 3), size / 2 + rng.normal(0, 3)
    ry, rx = 88 + rng.normal(0, 4), 74 + rng.normal(0, 4)
    d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    brain = d < 1.0
    img = np.zeros((size, size), np.float32)
    img[brain] = 0.35 + 0.05 * rng.standard_normal(brain.sum())
    img[(d > 0.86) & brain] += 0.25                       # skull-like bright rim
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    tumour = np.zeros((size, size), bool)
    if label != 2:
        if label == 0:       # glioma: large, irregular, interior
            ty, tx, s = cy - 15 + rng.normal(0, 8), cx + rng.normal(0, 14), 17
        elif label == 1:     # meningioma: peripheral, round
            ang = rng.uniform(0, 2 * np.pi)
            ty, tx, s = cy + 0.62 * ry * np.sin(ang), cx + 0.62 * rx * np.cos(ang), 11
        else:                # pituitary: small, central-low
            ty, tx, s = cy + 38 + rng.normal(0, 4), cx + rng.normal(0, 5), 8
        blob = np.exp(-(((yy - ty) / s) ** 2 + ((xx - tx) / s) ** 2) / 2)
        if label == 0:
            blob *= 1 + 0.3 * cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 3) * 6
            blob = np.clip(blob, 0, None)
        img = img + 0.45 * blob.astype(np.float32) * brain
        tumour = (blob > 0.5) & brain
    img = np.clip(img, 0, 1)
    return np.repeat(img[..., None], 3, axis=-1).astype(np.float32), tumour


def _response(img):
    g = img.mean(-1)
    brain = cv2.GaussianBlur((g > 0.08).astype(np.float32), (0, 0), 2) > 0.5
    # ignore the skull rim: use a heavily eroded brain core
    core = cv2.erode(brain.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    if core.sum() < 50:
        core = brain
    bg = np.median(g[core]) if core.any() else 0.3
    sm = cv2.GaussianBlur(g, (0, 0), 6)
    resp = (sm - bg) * core
    return resp, core


def fake_predict(x, noise: float = 0.0):
    """Fake 4-class classifier driven by the strongest bright blob in the brain core."""
    x = np.asarray(x, np.float32)
    out = []
    for img in x:
        resp, core = _response(img)
        r = float(resp.max())
        y, xx = np.unravel_index(np.argmax(resp), resp.shape)
        ys, xs = np.nonzero(core)
        cy, cx = ys.mean(), xs.mean()
        dist = np.hypot(y - cy, xx - cx) / (np.sqrt(core.sum() / np.pi) + 1e-6)
        tumour = 1 / (1 + np.exp(-45 * (r - 0.10)))
        w_men = 1 / (1 + np.exp(-12 * (dist - 0.65)))
        w_pit = (1 - w_men) * (1 / (1 + np.exp(-0.25 * ((y - cy) - 25))))
        w_gli = 1 - w_men - w_pit
        p = np.array([tumour * w_gli, tumour * w_men, 1 - tumour, tumour * w_pit]) + 1e-4
        out.append(p / p.sum())
    return np.array(out, np.float32)


def fake_cam_a(img, class_idx):
    resp, _ = _response(img)
    m = np.clip(resp, 0, None)
    return m / (m.max() + 1e-8)


def fake_cam_b(img, class_idx):
    m = fake_cam_a(img, class_idx)
    rng = np.random.default_rng(int(img.sum() * 1000) % (2 ** 31))
    m = cv2.GaussianBlur(m, (0, 0), 3) + 0.15 * cv2.GaussianBlur(rng.random(m.shape).astype(np.float32), (0, 0), 8)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def fake_attention(img):
    return fake_cam_a(img, 0) ** 0.5
