"""Tests for everything that does not need TensorFlow.  Run:  python -m pytest tests -q
(or just ``python tests/test_core_numpy.py``)."""
import json
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from synthetic import CLASSES, fake_attention, fake_cam_a, fake_cam_b, fake_predict, make_mri  # noqa: E402
from xai_brain.config import EvidenceConfig  # noqa: E402
from xai_brain.evaluate import evaluate_dataset, summarize_evaluation  # noqa: E402
from xai_brain.evidence import FEATURE_NAMES, counterfactual_tests, region_features  # noqa: E402
from xai_brain.metrics import agreement, deletion_insertion, stability  # noqa: E402
from xai_brain.report import explain_image, make_evidence_figure, save_report  # noqa: E402
from xai_brain.saliency import occlusion_map  # noqa: E402
from xai_brain.surrogate import FeatureSurrogate, class_profiles  # noqa: E402
from xai_brain.uncertainty import (apply_temperature, expected_calibration_error, fit_temperature,  # noqa: E402
                                   predictive_stats)
from xai_brain.utils import brain_mask, hotspot_mask  # noqa: E402

RNG = np.random.default_rng(0)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_fake_model_sane():
    preds = []
    for lab in range(4):
        for _ in range(5):
            img, _ = make_mri(lab, RNG)
            preds.append((lab, int(fake_predict(img[None])[0].argmax())))
    acc = np.mean([a == b for a, b in preds])
    print(f"fake model accuracy on synthetic set: {acc:.2f}")
    assert acc > 0.7


def test_brain_mask_and_hotspot():
    img, tum = make_mri(0, RNG)
    brain, ok = brain_mask(img)
    assert ok and 0.3 < brain.mean() < 0.8
    hot = hotspot_mask(fake_cam_a(img, 0))
    print("hotspot/tumour IoU", iou(hot, tum))
    assert iou(hot, tum) > 0.15 and (hot & tum).sum() > 0


def test_faithfulness_and_counterfactual():
    img, tum = make_mri(0, RNG)
    c = int(fake_predict(img[None])[0].argmax())
    good = deletion_insertion(img, fake_cam_a(img, c), fake_predict, c)
    bad = deletion_insertion(img, np.random.default_rng(1).random(img.shape[:2]).astype(np.float32), fake_predict, c)
    print("faithfulness good/random", good["faithfulness"], bad["faithfulness"])
    assert good["faithfulness"] > bad["faithfulness"] + 0.05
    brain, _ = brain_mask(img)
    hot = hotspot_mask(fake_cam_a(img, c))
    cf = counterfactual_tests(img, hot, brain, fake_predict, c, n_random=20)
    print({k: round(v, 3) for k, v in cf.items() if isinstance(v, float)})
    assert cf["drop"] > 0.3 and cf["p_value"] <= 0.1 and cf["n_controls"] >= 10


def test_region_features():
    img, tum = make_mri(1, RNG)
    brain, _ = brain_mask(img)
    sal = fake_cam_a(img, 1)
    out = region_features(img, hotspot_mask(sal), brain, sal)
    f = out["features"]
    assert set(FEATURE_NAMES) <= set(f)
    assert all(np.isfinite(list(f.values())))
    assert f["contrast_z"] > 1.0, f["contrast_z"]            # bright tumour
    assert f["in_brain_mass"] > 0.95
    print("meningioma-like features:", {k: round(f[k], 2) for k in ("dist_center", "depth", "area_frac", "contrast_z")})


def test_occlusion_and_agreement_and_stability():
    img, _ = make_mri(0, RNG)
    c = int(fake_predict(img[None])[0].argmax())
    occ = occlusion_map(img, fake_predict, c)
    assert occ.shape == img.shape[:2] and occ.max() == 1.0
    ag = agreement({"a": fake_cam_a(img, c), "b": fake_cam_b(img, c), "occ": occ})
    print("agreement", ag["mean_spearman"], ag["mean_iou"])
    assert ag["mean_spearman"] > 0.2
    st = stability(fake_cam_a, img, c, n=3)
    assert st["mean"] > 0.8


def test_calibration():
    rng = np.random.default_rng(0)
    n, k = 3000, 4
    logits = rng.normal(0, 1, (n, k))
    labels = np.array([rng.choice(k, p=np.exp(l) / np.exp(l).sum()) for l in logits])
    over = np.exp(logits * 3) / np.exp(logits * 3).sum(1, keepdims=True)    # over-confident model
    ece0 = expected_calibration_error(over, labels)["ece"]
    T = fit_temperature(over, labels)
    ece1 = expected_calibration_error(apply_temperature(over, T), labels)["ece"]
    print(f"ECE {ece0:.3f} -> {ece1:.3f} with T={T:.2f} (true 3.0)")
    assert T > 2.0 and ece1 < ece0 / 2
    s = predictive_stats(np.array([0.7, 0.2, 0.05, 0.05]))
    assert s["top1"] == 0 and abs(s["margin"] - 0.5) < 1e-6


def _explainers():
    return {"cam_a": fake_cam_a, "cam_b": fake_cam_b, "occlusion": lambda im, c: occlusion_map(im, fake_predict, c)}


def test_explain_image_end_to_end():
    img, _ = make_mri(1, RNG)
    cfg = EvidenceConfig(n_random_controls=20)
    res = explain_image(img, fake_predict, _explainers(), CLASSES, true_label=1, config=cfg,
                        attention_fn=fake_attention, temperature=1.2)
    print(res["text"])
    assert res["evidence_strength"]["label"] in {"Strong", "Moderate", "Weak"}
    assert len(res["checklist"]) >= 9
    with tempfile.TemporaryDirectory() as d:
        paths = save_report(res, os.path.join(d, "case"))
        assert os.path.getsize(paths["png"]) > 20000
        json.load(open(paths["json"]))
        import shutil
        shutil.copy(paths["png"], os.path.join(ROOT, "outputs", "_synthetic_example_card.png"))


def test_dataset_level_and_surrogate():
    imgs, labels = [], []
    for lab in range(4):
        for _ in range(10):
            im, _ = make_mri(lab, RNG)
            imgs.append(im); labels.append(lab)
    cfg = EvidenceConfig(n_random_controls=10, n_tta=4, n_stability=2)
    df, results = evaluate_dataset(imgs, labels, fake_predict, _explainers(), CLASSES, cfg, progress=False)
    assert len(df) == 40 and np.isfinite(df[FEATURE_NAMES].to_numpy()).all()
    summ = summarize_evaluation(df, CLASSES)
    print(summ["method_faithfulness"]); print(summ["correct_vs_wrong"])
    sur = FeatureSurrogate(CLASSES).fit(df)
    g = sur.global_summary()
    print("surrogate fidelity", g["fidelity_logreg"], "majority", g["majority_baseline"])
    print(g["permutation_importance"].head(5))
    assert g["fidelity_logreg"] > g["majority_baseline"]
    prof = class_profiles(df, CLASSES)
    res = explain_image(imgs[5], fake_predict, _explainers(), CLASSES, true_label=0, config=cfg,
                        profile=prof, surrogate=sur)
    assert "FEATURE ATTRIBUTION" in res["text"] and "typical features" in res["text"]
    print(res["text"][:1500])
    from xai_brain import global_analysis as ga
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as d:
        ga.plot_feature_importance(g["permutation_importance"], path=os.path.join(d, "a.png"))
        ga.plot_error_analysis(df, path=os.path.join(d, "b.png"))
        emb = df[FEATURE_NAMES].to_numpy()
        ga.plot_tsne(emb, df["label"], CLASSES, df["pred"], path=os.path.join(d, "c.png"))
        plt.close("all")


def test_surrogate_with_only_two_predicted_classes():
    """Regression: a weak model that predicts just 2 classes crashed global_summary (coef_ shape (1,F))."""
    import pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    df["pred"] = rng.choice([0, 2], size=60)
    df["label"] = df["pred"]
    sur = FeatureSurrogate(CLASSES).fit(df)
    g = sur.global_summary()
    assert g["coefficients"].shape == (2, len(FEATURE_NAMES))
    out = sur.explain({f: 0.5 for f in FEATURE_NAMES}, 2)
    assert "supporting" in out and sur.explain({f: 0.5 for f in FEATURE_NAMES}, 1)["supporting"] == []


def test_failing_explainer_is_isolated():
    img, _ = make_mri(0, RNG)
    ex = {"ok": fake_cam_a, "broken": lambda im, c: 1 / 0}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = explain_image(img, fake_predict, ex, CLASSES, config=EvidenceConfig(n_random_controls=5))
    assert "broken" in res["method_errors"] and "ok" in res["methods"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(f"\n=== {name}")
            fn()
    print("\nALL PASSED")
