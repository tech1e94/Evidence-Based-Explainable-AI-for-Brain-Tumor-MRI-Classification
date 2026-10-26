"""Interpretable surrogate: which measurable region features explain the CNN's decisions?

A multinomial logistic regression (+ a shallow decision tree for readable rules) is fitted
to predict the *CNN's predicted class* from the evidence-region features. Its fidelity
(cross-validated agreement with the CNN) is reported - if it is low, the surrogate is NOT
a trustworthy description of the CNN and the report says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .evidence import FEATURE_DESCRIPTIONS, FEATURE_NAMES


class FeatureSurrogate:
    def __init__(self, class_names, feature_names=FEATURE_NAMES):
        self.class_names = list(class_names)
        self.feature_names = list(feature_names)
        self.fitted = False

    # -----------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = "pred", seed: int = 0):
        X = df[self.feature_names].to_numpy(float)
        y = df[target_col].to_numpy(int)
        self.classes_present = np.unique(y)
        if len(self.classes_present) < 2:
            raise ValueError("Need at least two predicted classes to fit a surrogate.")
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.lr = LogisticRegression(max_iter=2000, C=0.5).fit(Xs, y)
        self.tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, len(y) // 40),
                                           random_state=seed).fit(X, y)
        n_splits = int(min(5, np.bincount(y)[np.bincount(y) > 0].min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            pred_lr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
                                        X, y, cv=cv)
            pred_tree = cross_val_predict(DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=max(5, len(y) // 40), random_state=seed), X, y, cv=cv)
            self.fidelity_lr = float((pred_lr == y).mean())
            self.fidelity_tree = float((pred_tree == y).mean())
        else:
            self.fidelity_lr = self.fidelity_tree = float("nan")
        self.majority = float(np.bincount(y).max() / len(y))
        perm = permutation_importance(self.lr, Xs, y, n_repeats=10, random_state=seed)
        self.importance = pd.Series(perm.importances_mean, index=self.feature_names).sort_values(ascending=False)
        self.fitted = True
        return self

    # -----------------------------------------------------------------------------
    def _coef(self) -> np.ndarray:
        """Coefficients as (n_classes_present, n_features). scikit-learn returns a single row when
        only two classes were predicted (it describes classes_[1]); expand it to two mirrored rows."""
        c = self.lr.coef_
        return np.vstack([-c[0], c[0]]) if c.shape[0] == 1 and len(self.lr.classes_) == 2 else c

    def global_summary(self) -> dict:
        names = [self.class_names[c] for c in self.lr.classes_]
        coef = pd.DataFrame(self._coef(), index=names, columns=self.feature_names)
        rules = export_text(self.tree, feature_names=self.feature_names)
        return {"fidelity_logreg": self.fidelity_lr, "fidelity_tree": self.fidelity_tree,
                "majority_baseline": self.majority, "permutation_importance": self.importance,
                "coefficients": coef, "tree_rules": rules}

    def explain(self, feats: dict, class_idx: int, top: int = 4) -> dict:
        """Per-feature contribution (standardised value x class coefficient) for one image."""
        if class_idx not in self.lr.classes_:
            return {"supporting": [], "opposing": [], "fidelity": self.fidelity_lr}
        row = int(np.where(self.lr.classes_ == class_idx)[0][0])
        x = self.scaler.transform(np.array([[feats[f] for f in self.feature_names]], float))[0]
        contrib = self._coef()[row] * x
        order = np.argsort(-contrib)
        mk = lambda i: {"feature": self.feature_names[i], "value": float(feats[self.feature_names[i]]),
                        "z": float(x[i]), "contribution": float(contrib[i]),
                        "meaning": FEATURE_DESCRIPTIONS[self.feature_names[i]]}
        return {"supporting": [mk(i) for i in order[:top] if contrib[i] > 0],
                "opposing": [mk(i) for i in order[::-1][:top] if contrib[i] < 0],
                "fidelity": self.fidelity_lr}


def class_profiles(df: pd.DataFrame, class_names, feature_names=FEATURE_NAMES, class_col: str = "pred",
                   correct_only: bool = True) -> dict:
    """Mean / std of every feature per class (from correctly classified cases by default)."""
    d = df[df["pred"] == df["label"]] if (correct_only and "label" in df) else df
    prof = {}
    for c, name in enumerate(class_names):
        sub = d[d[class_col] == c]
        if len(sub) >= 3:
            prof[name] = {f: (float(sub[f].mean()), float(sub[f].std() + 1e-6)) for f in feature_names}
    return prof


_TRIVIAL = {"n_components"}


def compare_to_profile(feats: dict, profile: dict, class_name: str, feature_names=FEATURE_NAMES) -> dict:
    """z-score of this image's features against typical cases of the predicted class.

    Near-constant features (class std < 1e-3) and bookkeeping features are skipped - they would
    otherwise always look "perfectly typical".
    """
    if class_name not in profile:
        return {"typical": [], "unusual": []}
    use = [f for f in feature_names if f not in _TRIVIAL and profile[class_name][f][1] > 1e-3]
    z = {f: (feats[f] - profile[class_name][f][0]) / profile[class_name][f][1] for f in use}
    ordered = sorted(z.items(), key=lambda kv: abs(kv[1]))
    mk = lambda f, v: {"feature": f, "z": float(v), "value": float(feats[f]),
                       "typical_mean": profile[class_name][f][0], "meaning": FEATURE_DESCRIPTIONS[f]}
    return {"typical": [mk(f, v) for f, v in ordered[:3]],
            "unusual": [mk(f, v) for f, v in ordered[::-1][:3] if abs(v) > 2.0]}
