"""Dataset-level analysis: run the evidence report over many images and summarise.

Answers the questions a single image cannot:
  * Which explanation method is the most faithful on average?
  * Do the explanation-quality metrics tell us when the model is wrong?
  * Which region features separate the classes (surrogate / class profiles)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EvidenceConfig
from .evidence import FEATURE_NAMES
from .report import explain_image


def result_to_row(res: dict, idx: int, label: int) -> dict:
    row = {
        "idx": idx, "label": int(label), "pred": res["pred"], "correct": int(res["pred"] == label),
        "confidence": res["confidence"], "margin": res["margin"], "entropy_norm": res["entropy_norm"],
        "tta_agreement": res["tta"]["agreement"],
        "faith_consensus": res["consensus"]["faithfulness"],
        "agreement": res["agreement"]["mean_spearman"] if res["agreement"] else np.nan,
        "necessity_drop": res["causal"]["drop"], "specificity_p": res["causal"]["p_value"],
        "sufficiency": res["causal"]["sufficiency"],
        "stability": res["stability"]["mean"] if res["stability"] else np.nan,
        "evidence_share": res["evidence_strength"]["share"], "evidence_label": res["evidence_strength"]["label"],
    }
    for n, m in res["methods"].items():
        row[f"faith_{n}"] = m["faithfulness"]
    row.update({k: res["features"][k] for k in FEATURE_NAMES})
    return row


def evaluate_dataset(images, labels, predict_fn, explainers, class_names, config: EvidenceConfig | None = None,
                     progress: bool = True, **explain_kwargs):
    """Run ``explain_image`` on every image (no figures). Returns ``(DataFrame, results_list)``."""
    rows, results = [], []
    it = enumerate(zip(images, labels))
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(list(it), desc="explaining")
        except ImportError:
            pass
    for i, (img, y) in it:
        res = explain_image(img, predict_fn, explainers, class_names, true_label=int(y), config=config,
                            return_arrays=False, **explain_kwargs)
        results.append(res)
        rows.append(result_to_row(res, i, y))
    return pd.DataFrame(rows), results


def summarize_evaluation(df: pd.DataFrame, class_names) -> dict:
    """Aggregate tables used in the README / notebook."""
    from sklearn.metrics import roc_auc_score
    out = {}
    method_cols = [c for c in df.columns if c.startswith("faith_") and c != "faith_consensus"]
    out["method_faithfulness"] = (df[method_cols + ["faith_consensus"]].mean()
                                  .rename(lambda s: s.replace("faith_", "")).sort_values(ascending=False))
    metric_cols = ["confidence", "margin", "tta_agreement", "faith_consensus", "agreement", "necessity_drop",
                   "sufficiency", "stability", "in_brain_mass", "rim_mass", "evidence_share"]
    out["correct_vs_wrong"] = df.groupby("correct")[metric_cols].mean().rename(index={0: "wrong", 1: "correct"})
    # can an explanation metric flag mistakes?  AUROC for predicting 'model is RIGHT'
    aur = {}
    if df["correct"].nunique() == 2:
        for col in metric_cols:
            d = df[[col, "correct"]].dropna()
            if d["correct"].nunique() == 2 and d[col].nunique() > 1:
                aur[col] = float(roc_auc_score(d["correct"], d[col]))
    out["error_detection_auroc"] = pd.Series(aur).sort_values(ascending=False)
    out["per_class"] = df.assign(cls=df["pred"].map(dict(enumerate(class_names)))).groupby("cls")[
        ["confidence", "faith_consensus", "necessity_drop", "area_frac", "contrast_z", "depth", "evidence_share"]].mean()
    out["evidence_label_counts"] = df["evidence_label"].value_counts()
    return out
