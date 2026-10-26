"""Global (model-level) explanations and plots."""
from __future__ import annotations

import numpy as np

from .utils import overlay


# ---------------------------------------------------------------------------------------
# Example-based explanation: nearest reference cases in the model's feature space
# ---------------------------------------------------------------------------------------
def make_neighbour_fn(embed_fn, ref_emb: np.ndarray, ref_labels: np.ndarray, ref_paths=None, k: int = 5):
    """``fn(img) -> [{index, label, similarity, path}]`` using cosine similarity."""
    ref = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-8)

    def fn(img):
        e = np.asarray(embed_fn(img[None]))[0]
        e = e / (np.linalg.norm(e) + 1e-8)
        sims = ref @ e
        idx = np.argsort(-sims)[:k]
        return [{"index": int(i), "label": int(ref_labels[i]), "similarity": float(sims[i]),
                 "path": None if ref_paths is None else str(ref_paths[i])} for i in idx]
    return fn


def class_mean_maps(images, preds, explainer, class_names, max_per_class: int = 25):
    """Average explanation map per predicted class (what does the model look at, on average?)."""
    out = {}
    for c, name in enumerate(class_names):
        idx = np.where(np.asarray(preds) == c)[0][:max_per_class]
        if len(idx):
            out[name] = np.mean([explainer(images[i], c) for i in idx], axis=0)
    return out


# ---------------------------------------------------------------------------------------
# Plots (matplotlib)
# ---------------------------------------------------------------------------------------
def plot_class_mean_maps(mean_maps: dict, ref_images: dict | None = None, path: str | None = None):
    import matplotlib.pyplot as plt
    n = len(mean_maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.3), squeeze=False)
    for ax, (name, m) in zip(axes[0], mean_maps.items()):
        base = ref_images[name] if ref_images and name in ref_images else np.full(m.shape + (3,), 0.15)
        ax.imshow(overlay(base, m / (m.max() + 1e-8), alpha=0.6)); ax.axis("off"); ax.set_title(name)
    fig.suptitle("Average evidence map per predicted class", y=1.02)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def plot_reliability(cal: dict, cal_after: dict | None = None, path: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for res, lab, col in ((cal, "before", "tab:red"), (cal_after, "after temp. scaling", "tab:green")):
        if res and res["bins"]:
            ax.plot([b["confidence"] for b in res["bins"]], [b["accuracy"] for b in res["bins"]], "o-", color=col,
                    label=f"{lab} (ECE {res['ece']:.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(); ax.set_title("Reliability diagram")
    ax.grid(alpha=0.3)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def plot_tsne(emb: np.ndarray, labels, class_names, preds=None, path: str | None = None, seed: int = 0):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    perp = float(min(30, max(5, (len(emb) - 1) / 3)))
    xy = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca").fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        s = np.asarray(labels) == c
        ax.scatter(xy[s, 0], xy[s, 1], s=14, label=name, alpha=0.8)
    if preds is not None:
        wrong = np.asarray(preds) != np.asarray(labels)
        ax.scatter(xy[wrong, 0], xy[wrong, 1], s=60, facecolors="none", edgecolors="k", linewidths=1,
                   label="misclassified")
    ax.legend(); ax.set_title("t-SNE of the model's feature space")
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def plot_feature_importance(importance, title="Surrogate feature importance (permutation)", path=None, top=12):
    import matplotlib.pyplot as plt
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="tab:blue")
    ax.set_xlabel("drop in surrogate accuracy when feature is shuffled"); ax.set_title(title)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def plot_error_analysis(df, path: str | None = None):
    """Distribution of key explanation metrics for correct vs wrong predictions."""
    import matplotlib.pyplot as plt
    cols = [c for c in ("confidence", "faith_consensus", "necessity_drop", "agreement", "in_brain_mass", "rim_mass")
            if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3.8))
    for ax, col in zip(np.atleast_1d(axes), cols):
        data = [df.loc[df["correct"] == 1, col].dropna(), df.loc[df["correct"] == 0, col].dropna()]
        try:
            ax.boxplot(data, tick_labels=["correct", "wrong"], showmeans=True)
        except TypeError:          # matplotlib < 3.9
            ax.boxplot(data, labels=["correct", "wrong"], showmeans=True)
        ax.set_title(col, fontsize=10)
    fig.suptitle("Do explanation metrics differ when the model is wrong?", y=1.03)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig
