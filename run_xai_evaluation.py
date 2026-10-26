"""Dataset-level XAI evaluation. Produces everything in ``outputs/``:

    python run_xai_evaluation.py --model models/improved_cnn.keras --data archive --per-class 25

* calibration (temperature fitted on the validation split) + reliability diagram
* evidence metrics for N images per class  -> xai_metrics.csv, summary tables
* class feature profiles + interpretable surrogate -> class_profiles.json, surrogate.joblib
* parameter-randomisation sanity check of Grad-CAM
* one evidence card per class (outputs/cards/)
"""
import argparse
import json
import os

import joblib
import numpy as np

from xai_brain import global_analysis as ga
from xai_brain.config import CATEGORIES, EvidenceConfig
from xai_brain.data import find_split_dir, list_images, load_images, sample_per_class, train_val_split
from xai_brain.evaluate import evaluate_dataset, summarize_evaluation
from xai_brain.report import explain_image, save_report
from xai_brain.surrogate import FeatureSurrogate, class_profiles
from xai_brain.uncertainty import apply_temperature, expected_calibration_error, fit_temperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="archive")
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--methods", default="gradcam,gradcampp,ig,occlusion",
                    help="add scorecam for a gradient-free check (slow on CPU)")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cards"), exist_ok=True)

    from xai_brain.channels import channel_evidence
    from xai_brain.model import get_embeddings, load_model, make_predict_fn
    from xai_brain.saliency import build_explainers, parameter_randomization_test

    model = load_model(a.model)
    predict = make_predict_fn(model)
    explainers, ge = build_explainers(model, predict, tuple(a.methods.split(",")))
    cfg = EvidenceConfig()

    # --- calibration on the validation split ------------------------------------------------
    paths, labels = list_images(find_split_dir(a.data, "Training"))
    _, (va_p, va_y) = train_val_split(paths, labels, 0.1)
    vp, vy = sample_per_class(va_p, va_y, 100)
    val_probs = predict(load_images(vp))
    T = fit_temperature(val_probs, vy)
    before = expected_calibration_error(val_probs, vy)
    after = expected_calibration_error(apply_temperature(val_probs, T), vy)
    print(f"temperature {T:.2f}: ECE {before['ece']:.3f} -> {after['ece']:.3f}")
    ga.plot_reliability(before, after, os.path.join(a.out, "reliability.png"))

    # --- explain a class-balanced test sample --------------------------------------------------
    te_p, te_y = list_images(find_split_dir(a.data, "Testing"))
    sp, sy = sample_per_class(te_p, te_y, a.per_class)
    X = load_images(sp)
    sanity = parameter_randomization_test(model, X[:10], predict(X[:10]).argmax(1))
    print("randomisation sanity:", {k: v for k, v in sanity.items() if k.startswith("mean") or k == "passed"})
    df, _ = evaluate_dataset(X, sy, predict, explainers, CATEGORIES, cfg, temperature=T, sanity=sanity)
    df["path"] = sp
    df.to_csv(os.path.join(a.out, "xai_metrics.csv"), index=False)
    summ = summarize_evaluation(df, CATEGORIES)
    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        for k, v in summ.items():
            fh.write(f"== {k}\n{v}\n\n")
    print(summ["method_faithfulness"]); print(summ["error_detection_auroc"])
    ga.plot_error_analysis(df, os.path.join(a.out, "error_analysis.png"))

    # --- profiles + surrogate -------------------------------------------------------------------------
    prof = class_profiles(df, CATEGORIES)
    json.dump(prof, open(os.path.join(a.out, "class_profiles.json"), "w"), indent=1)
    sur = FeatureSurrogate(CATEGORIES).fit(df)
    joblib.dump(sur, os.path.join(a.out, "surrogate.joblib"))
    g = sur.global_summary()
    print(f"surrogate fidelity {g['fidelity_logreg']:.2f} (majority baseline {g['majority_baseline']:.2f})")
    open(os.path.join(a.out, "surrogate_rules.txt"), "w").write(g["tree_rules"])
    g["coefficients"].to_csv(os.path.join(a.out, "surrogate_coefficients.csv"))
    ga.plot_feature_importance(g["permutation_importance"], path=os.path.join(a.out, "feature_importance.png"))

    # --- global views --------------------------------------------------------------------------------------
    preds = predict(X).argmax(1)
    primary = a.methods.split(",")[0]
    mm = ga.class_mean_maps(X, preds, explainers[primary], CATEGORIES)
    ga.plot_class_mean_maps(mm, path=os.path.join(a.out, "class_mean_maps.png"))
    ga.plot_tsne(get_embeddings(model, X), sy, CATEGORIES, preds, os.path.join(a.out, "tsne.png"))

    # --- one full card per class ------------------------------------------------------------------------------
    tr_p, tr_y = sample_per_class(paths, labels, 40)
    ref_emb = get_embeddings(model, load_images(tr_p))
    nb = ga.make_neighbour_fn(lambda x: get_embeddings(model, x), ref_emb, tr_y, tr_p)
    for c, name in enumerate(CATEGORIES):
        i = int(np.where(sy == c)[0][0])
        res = explain_image(X[i], predict, explainers, CATEGORIES, true_label=int(sy[i]), config=cfg,
                            profile=prof, surrogate=sur, temperature=T, attention_fn=ge.attention if ge.gate_model else None,
                            channel_fn=lambda im, k: channel_evidence(ge, im, k, predict), neighbour_fn=nb, sanity=sanity)
        save_report(res, os.path.join(a.out, "cards", f"{name}_example"))
    print("done ->", a.out)


if __name__ == "__main__":
    main()
