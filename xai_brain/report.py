"""Orchestrator: one call -> an evidence-backed explanation for one MRI slice.

    result = explain_image(img, predict_fn, explainers, CATEGORIES, ...)
    print(result["text"])                 # numbers-backed narrative
    make_evidence_figure(result, "card.png")
    save_report(result, "outputs/case_001")

Every sentence in ``result["text"]`` is generated from a measured number stored in
``result`` - nothing is free-form. The pass/fail ``checklist`` replaces the old
single "trust" heuristic with transparent, individually inspectable criteria.
"""
from __future__ import annotations

import json
import textwrap
import warnings

import cv2
import numpy as np

from .config import EvidenceConfig
from .evidence import (FEATURE_DESCRIPTIONS, counterfactual_tests, describe_features, region_features)
from .metrics import agreement, deletion_insertion, map_correlation, stability, top_iou
from .saliency import METHOD_LABELS
from .surrogate import compare_to_profile
from .uncertainty import predictive_stats, apply_temperature, tta_uncertainty, sample_uncertainty
from .utils import (brain_mask, hotspot_mask, minmax, overlay, rank_normalize, to_float01)


# ======================================================================================
def explain_image(img, predict_fn, explainers: dict, class_names, *, true_label: int | None = None,
                  config: EvidenceConfig | None = None, profile: dict | None = None, surrogate=None,
                  temperature: float = 1.0, attention_fn=None, channel_fn=None, neighbour_fn=None,
                  mc_fn=None, sanity: dict | None = None, class_idx: int | None = None,
                  stability_method: str | None = None, return_arrays: bool = True) -> dict:
    """Build the full evidence report for one image (see module docstring).

    Parameters
    ----------
    explainers      dict name -> fn(img, class_idx) -> map in [0, 1]
    profile         class feature profile from ``surrogate.class_profiles`` (optional)
    surrogate       fitted ``FeatureSurrogate`` (optional)
    temperature     calibration temperature from ``uncertainty.fit_temperature``
    attention_fn    fn(img) -> built-in attention map (optional)
    channel_fn      fn(img, class_idx) -> dict from ``channels.channel_evidence`` (optional)
    neighbour_fn    fn(img) -> list of {"label", "similarity", "path"/"index"} (optional)
    mc_fn           fn(img) -> MC-dropout samples [n, K] (optional)
    sanity          dict from ``parameter_randomization_test`` (optional, dataset-level)
    class_idx       class to explain (default = predicted class)
    """
    cfg = config or EvidenceConfig()
    img = to_float01(img)
    K = len(class_names)

    # ---- 1. prediction + uncertainty --------------------------------------------------
    raw = np.asarray(predict_fn(img[None]))[0]
    cal = apply_temperature(raw[None], temperature)[0] if temperature != 1.0 else raw
    st = predictive_stats(cal)
    c = int(st["top1"]) if class_idx is None else int(class_idx)
    tta = tta_uncertainty(img, predict_fn, base_class=int(st["top1"]), n=cfg.n_tta, seed=cfg.random_seed)
    mc = sample_uncertainty(mc_fn(img), int(st["top1"])) if mc_fn is not None else None
    res = {
        "class_names": list(class_names), "true_label": true_label,
        "pred": int(st["top1"]), "explained_class": c, "second": int(st["top2"]),
        "probs_raw": raw.tolist(), "probs_calibrated": cal.tolist(), "temperature": float(temperature),
        "confidence": st["p1"], "margin": st["margin"], "entropy_norm": st["entropy_norm"],
        "tta": tta, "mc_dropout": mc,
    }

    # ---- 2. saliency maps + faithfulness -------------------------------------------------
    brain, brain_ok = brain_mask(img)
    res["brain_mask_ok"] = bool(brain_ok)
    maps, methods, errors = {}, {}, {}
    for name, fn in explainers.items():
        try:
            m = np.asarray(fn(img, c), dtype=np.float32)
            maps[name] = minmax(m)
            di = deletion_insertion(img, maps[name], predict_fn, c, cfg.curve_steps, cfg.fill_mode)
            methods[name] = {"label": METHOD_LABELS.get(name, name), **{k: di[k] for k in
                             ("auc_deletion", "auc_insertion", "faithfulness")}, "curves": {
                             "fractions": di["fractions"], "deletion": di["deletion"], "insertion": di["insertion"]}}
        except Exception as exc:                                    # one failing method must not kill the report
            errors[name] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"explainer '{name}' failed: {errors[name]}")
    if not maps:
        raise RuntimeError(f"all explainers failed: {errors}")
    res["methods"], res["method_errors"] = methods, errors

    good = [n for n in maps if methods[n]["faithfulness"] > 0] or list(maps)
    primary = max(maps, key=lambda n: methods[n]["faithfulness"])
    consensus = np.mean([maps[n] for n in good], axis=0)      # mean of min-max maps keeps the map sparse
    consensus = minmax(cv2.GaussianBlur(consensus.astype(np.float32), (0, 0), cfg.consensus_smooth_sigma))
    di_c = deletion_insertion(img, consensus, predict_fn, c, cfg.curve_steps, cfg.fill_mode)
    res["consensus"] = {"methods_used": good, "primary": primary,
                        **{k: di_c[k] for k in ("auc_deletion", "auc_insertion", "faithfulness")},
                        "curves": {k: di_c[k] for k in ("fractions", "deletion", "insertion")}}
    agr = agreement({n: maps[n] for n in good}) if len(good) > 1 else None
    res["agreement"] = ({"mean_spearman": agr["mean_spearman"], "mean_iou": agr["mean_iou"], "names": agr["names"],
                         "spearman": agr["spearman"].tolist(), "iou": agr["iou"].tolist()} if agr else None)

    # ---- 3. stability of the explanation ------------------------------------------------------
    stab_name = stability_method or primary
    stab = None
    if stab_name in explainers and stab_name != "lime" and cfg.n_stability > 0:
        try:
            stab = stability(explainers[stab_name], img, c, base_map=maps.get(stab_name),
                             n=cfg.n_stability, sigma=cfg.stability_sigma, seed=cfg.random_seed)
            stab["method"] = stab_name
        except Exception as exc:
            warnings.warn(f"stability check failed: {exc}")
    res["stability"] = stab

    # ---- 4. evidence region, causal tests, features ---------------------------------------------------
    hot = hotspot_mask(consensus, cfg.hotspot_top_frac, cfg.hotspot_min_area, cfg.hotspot_max_components)
    if not hot.any():
        hot = consensus >= np.quantile(consensus, 0.9)
    cf = counterfactual_tests(img, hot, brain, predict_fn, c, cfg.n_random_controls, cfg.fill_mode, cfg.random_seed)
    rf = region_features(img, hot, brain, saliency=consensus)
    feats, region = rf["features"], rf["region"]
    res["causal"] = {k: v for k, v in cf.items() if k != "arrays"}
    res["features"] = feats
    res["feature_words"] = describe_features(feats)
    res["feature_vs_class_profile"] = (compare_to_profile(feats, profile, class_names[c])
                                       if profile else None)
    res["surrogate"] = surrogate.explain(feats, c) if surrogate is not None and surrogate.fitted else None
    if surrogate is not None and surrogate.fitted:
        res["surrogate_fidelity"] = {"logreg": surrogate.fidelity_lr, "majority_baseline": surrogate.majority}

    # contrastive: why this class and not the runner-up?
    if st["top1"] == c and K > 1:
        second = int(st["top2"])
        contrast = {"second": second, "p_second": float(cal[second]),
                    "p_second_after_removal": float(cf["probs_removed"][second]),
                    "class_after_removal": int(cf["class_after_removal"])}
        try:
            m2 = np.asarray(explainers[primary](img, second), dtype=np.float32)
            contrast["map_correlation"] = map_correlation(maps[primary], m2)
            contrast["arrays_map"] = minmax(m2)
        except Exception as exc:
            warnings.warn(f"contrastive map failed: {exc}")
        res["contrast"] = {k: v for k, v in contrast.items() if k != "arrays_map"}
    else:
        contrast = None
        res["contrast"] = None

    # ---- 5. optional extras ---------------------------------------------------------------------------------
    attn = None
    if attention_fn is not None:
        try:
            attn = minmax(np.asarray(attention_fn(img)))
            res["attention"] = {"spearman_with_consensus": map_correlation(attn, consensus),
                                "iou_with_consensus": top_iou(attn, consensus)}
        except Exception as exc:
            warnings.warn(f"attention map failed: {exc}")
    res["channels"] = None
    ch_arrays = None
    if channel_fn is not None:
        try:
            ch = channel_fn(img, c)
            ch_arrays = ch.pop("arrays", None)
            res["channels"] = ch
        except Exception as exc:
            warnings.warn(f"channel evidence failed: {exc}")
    res["neighbours"] = neighbour_fn(img) if neighbour_fn is not None else None
    res["sanity"] = sanity

    # ---- 6. checklist + text ---------------------------------------------------------------------------------------
    res["checklist"] = build_checklist(res, cfg)
    res["evidence_strength"] = summarise_strength(res["checklist"], cfg)
    res["text"] = compose_text(res, cfg)

    if return_arrays:
        res["arrays"] = {"img": img, "maps": maps, "consensus": consensus, "hotspot": hot, "region": region,
                         "brain": brain, "removed": cf["arrays"]["removed"], "kept": cf["arrays"]["kept"],
                         "attention": attn, "contrast_map": (contrast or {}).get("arrays_map"),
                         "channel_maps": (ch_arrays or {}).get("maps")}
    return res


# ======================================================================================
# Checklist
# ======================================================================================
def _chk(name, value, passed, rule, why):
    return {"name": name, "value": value, "passed": passed, "rule": rule, "why": why}


def build_checklist(r: dict, cfg: EvidenceConfig) -> list:
    L = []
    f, cz = r["features"], r["causal"]
    L.append(_chk("Calibrated confidence", f"{r['confidence']:.0%}", r["confidence"] >= cfg.min_confidence,
                  f">= {cfg.min_confidence:.0%}", "model is reasonably sure"))
    L.append(_chk("Margin over runner-up", f"{r['margin']:.2f}", r["margin"] >= cfg.min_margin,
                  f">= {cfg.min_margin:.2f}", "decision is not a near tie"))
    L.append(_chk("Prediction stable under augmentation", f"{r['tta']['agreement']:.0%}",
                  r["tta"]["agreement"] >= cfg.min_tta_agreement, f">= {cfg.min_tta_agreement:.0%}",
                  "class does not flip with small shifts/rotations/brightness"))
    L.append(_chk("Explanation is faithful (insertion - deletion AUC)", f"{r['consensus']['faithfulness']:+.2f}",
                  r["consensus"]["faithfulness"] >= cfg.min_faithfulness, f">= {cfg.min_faithfulness:+.2f}",
                  "revealing highlighted pixels raises, deleting them lowers, P(class)"))
    flipped = cz["class_after_removal"] != r["explained_class"]
    L.append(_chk("Region is NECESSARY (removal drops P(class))",
                  f"-{cz['drop']:.2f}" + (" & class flips" if flipped else ""),
                  (cz["drop"] >= cfg.min_necessity_drop) or flipped, f"drop >= {cfg.min_necessity_drop:.2f} or flip",
                  "the model needs this region for its answer"))
    if cz["n_controls"] >= 5:
        L.append(_chk("Region is SPECIFIC (vs random same-shape regions)", f"p = {cz['p_value']:.3f}",
                      cz["p_value"] <= cfg.max_control_p, f"p <= {cfg.max_control_p}",
                      "removing random regions does not hurt as much"))
    L.append(_chk("Region is SUFFICIENT (keep only region)", f"{cz['sufficiency']:.0%} of original",
                  cz["sufficiency"] >= cfg.min_sufficiency, f">= {cfg.min_sufficiency:.0%}",
                  "the region alone largely reproduces the prediction"))
    if r["agreement"] is not None:
        L.append(_chk("Explanation methods agree (mean Spearman)", f"{r['agreement']['mean_spearman']:.2f}",
                      r["agreement"]["mean_spearman"] >= cfg.min_agreement, f">= {cfg.min_agreement:.2f}",
                      "independent methods point at the same place"))
    L.append(_chk("Evidence lies inside the brain", f"{f['in_brain_mass']:.0%}",
                  f["in_brain_mass"] >= cfg.min_in_brain_mass, f">= {cfg.min_in_brain_mass:.0%}",
                  "not driven by background / scanner artefacts"))
    L.append(_chk("Evidence is not just the skull edge", f"{f['rim_mass']:.0%} on rim",
                  f["rim_mass"] <= cfg.max_rim_mass, f"<= {cfg.max_rim_mass:.0%}",
                  "rim-dominated maps often indicate shortcut learning"))
    if r["stability"] is not None:
        L.append(_chk("Explanation stable under tiny noise", f"{r['stability']['mean']:.2f}",
                      r["stability"]["mean"] >= cfg.min_stability, f">= {cfg.min_stability:.2f}",
                      "map does not change when the image barely changes"))
    if r.get("sanity"):
        L.append(_chk("Explanation depends on learned weights (randomisation test)",
                      f"|rho| = {r['sanity']['mean_abs_rho_full_random']:.2f}", bool(r["sanity"]["passed"]),
                      "|rho| < 0.5", "maps change when weights are randomised (dataset-level test)"))
    return L


def summarise_strength(checklist: list, cfg: EvidenceConfig) -> dict:
    app = [c for c in checklist if c["passed"] is not None]
    n_pass = sum(bool(c["passed"]) for c in app)
    share = n_pass / max(len(app), 1)
    label = "Strong" if share >= cfg.strong_cut else "Moderate" if share >= cfg.moderate_cut else "Weak"
    return {"label": label, "passed": n_pass, "total": len(app), "share": share,
            "failed": [c["name"] for c in app if not c["passed"]]}


# ======================================================================================
# Narrative
# ======================================================================================
def compose_text(r: dict, cfg: EvidenceConfig) -> str:
    names, c = r["class_names"], r["explained_class"]
    pred, sec = names[r["pred"]], names[r["second"]]
    cz, f, w = r["causal"], r["features"], r["feature_words"]
    S = []

    s = (f"PREDICTION - {pred}. Calibrated confidence {r['confidence']:.0%} "
         f"(raw {r['probs_raw'][r['pred']]:.0%}), margin {r['margin']:.2f} over '{sec}' "
         f"({r['probs_calibrated'][r['second']]:.0%}). Under {r['tta']['n']} small test-time changes the same class "
         f"was returned {r['tta']['agreement']:.0%} of the time.")
    if r["mc_dropout"]:
        s += (f" MC-dropout epistemic uncertainty (mutual information) = {r['mc_dropout']['mutual_information']:.3f}.")
    if r["true_label"] is not None:
        ok = r["true_label"] == r["pred"]
        s += f" Ground truth: {names[r['true_label']]} -> the prediction is {'CORRECT' if ok else 'WRONG'}."
    S.append(s)

    n_m = len(r["consensus"]["methods_used"])
    agr = (f"; the methods agree with Spearman rho = {r['agreement']['mean_spearman']:.2f}" if r["agreement"] else "")
    S.append(f"WHERE - Combining {n_m} explanation method(s){agr}, the strongest evidence for '{names[c]}' forms "
             f"{int(f['n_components'])} region(s) covering {f['area_frac']:.1%} of the brain ({w['size']}), located "
             f"{w['location']}. {f['in_brain_mass']:.0%} of the explanation lies inside the brain and "
             f"{f['rim_mass']:.0%} on its outer rim. Best single method by faithfulness: "
             f"{r['methods'][r['consensus']['primary']]['label']} "
             f"(insertion-deletion = {r['methods'][r['consensus']['primary']]['faithfulness']:+.2f}).")

    s = (f"WHICH FEATURES - Inside this region the tissue is {w['intensity']}; the region is {w['shape']} with "
         f"{w['edge']}, {w['texture']} texture (heterogeneity {f['heterogeneity']:.2f}, entropy "
         f"{f['texture_entropy']:.2f} bits) and {w['asymmetry']}.")
    prof = r.get("feature_vs_class_profile")
    if prof and prof["typical"]:
        typ = ", ".join(f"{x['feature']} (z={x['z']:+.1f})" for x in prof["typical"])
        s += f" Compared with correctly classified '{names[c]}' cases in your data, the most typical features are {typ}"
        if prof["unusual"]:
            s += "; unusual: " + ", ".join(f"{x['feature']} (z={x['z']:+.1f})" for x in prof["unusual"])
        s += "."
    S.append(s)

    sg = r.get("surrogate")
    if sg and (sg["supporting"] or sg["opposing"]):
        fid = r["surrogate_fidelity"]
        if fid["logreg"] >= fid["majority_baseline"] + 0.10:
            sup = ", ".join(f"{x['feature']}={x['value']:.2f} (+{x['contribution']:.2f})" for x in sg["supporting"])
            opp = ", ".join(f"{x['feature']}={x['value']:.2f} ({x['contribution']:.2f})" for x in sg["opposing"])
            S.append(f"FEATURE ATTRIBUTION - An interpretable surrogate (agrees with the CNN {fid['logreg']:.0%} of the "
                     f"time vs {fid['majority_baseline']:.0%} for always guessing the majority class) credits "
                     f"{'these features for the prediction: ' + sup if sup else 'no strongly supporting feature'}"
                     f"{'; against it: ' + opp if opp else ''}.")
        else:
            S.append(f"FEATURE ATTRIBUTION - The feature surrogate only agrees with the CNN {fid['logreg']:.0%} of the "
                     f"time (majority baseline {fid['majority_baseline']:.0%}), so its attribution is NOT reported as reliable.")

    ctrl = (f"random same-shape regions lower it by only {cz['control_mean_drop']:.2f} on average "
            f"(empirical p = {cz['p_value']:.3f}, n = {cz['n_controls']})" if cz["n_controls"] else
            "no valid random control regions could be placed")
    flip = (f" and the prediction changes to '{names[cz['class_after_removal']]}'"
            if cz["class_after_removal"] != c else " but the predicted class stays the same")
    S.append(f"CAUSAL CHECK - Removing that region lowers P({names[c]}) from {cz['p_orig']:.2f} to {cz['p_removed']:.2f} "
             f"(-{cz['drop']:.2f}){flip}; {ctrl}. Keeping only the region retains {cz['sufficiency']:.0%} of the "
             f"original probability.")

    if r.get("channels"):
        ch = r["channels"]
        s = (f"INTERNAL FEATURES - Feature channels {ch['channels']} of the last convolution contribute most to the "
             f"class score.")
        if "drop_top_k" in ch:
            s += (f" Switching them off lowers P by {ch['drop_top_k']:.2f} versus {ch['random_channel_drop_mean']:.2f} "
                  f"for the same number of random channels.")
        S.append(s)

    ct = r.get("contrast")
    if ct:
        s = (f"WHY NOT '{names[ct['second']]}'? - Its probability is {ct['p_second']:.0%}; after the evidence region is "
             f"removed it becomes {ct['p_second_after_removal']:.0%}")
        if "map_correlation" in ct:
            s += (f". The evidence map for '{names[ct['second']]}' correlates {ct['map_correlation']:+.2f} with the one "
                  f"for '{pred}' ({'similar' if ct['map_correlation'] > 0.5 else 'different'} regions are used)")
        S.append(s + ".")

    if r.get("neighbours"):
        nb = r["neighbours"]
        s = "SIMILAR CASES - Closest reference images in the model's feature space: " + ", ".join(
            f"{names[n['label']]} ({n['similarity']:.2f})" for n in nb)
        same = np.mean([n["label"] == r["pred"] for n in nb])
        S.append(s + f". {same:.0%} share the predicted class.")

    ev = r["evidence_strength"]
    verdict = (f"EVIDENCE STRENGTH - {ev['label'].upper()} ({ev['passed']}/{ev['total']} checks passed)")
    if ev["failed"]:
        verdict += "; not satisfied: " + "; ".join(ev["failed"])
    S.append(verdict + ".")
    S.append("CAVEATS - The highlighted region is where the MODEL looked, not a radiologist's tumour outline; intensity "
             "words describe the image, not a diagnosis; thresholds are heuristic conventions. This is research "
             "software, not a medical device.")
    return "\n\n".join(S)


# ======================================================================================
# Figure
# ======================================================================================
def make_evidence_figure(r: dict, path: str | None = None, dpi: int = 110):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    A = r["arrays"]
    img, names, c = A["img"], r["class_names"], r["explained_class"]
    cz = r["causal"]
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(5, 6, figure=fig, height_ratios=[1, 1, 1, 1.05, 1.5], hspace=0.38, wspace=0.12,
                  top=0.95, bottom=0.02, left=0.04, right=0.98)

    def show(ax, im, title, fs=10):
        ax.imshow(im)
        ax.set_title(title, fontsize=fs)
        ax.axis("off")

    # row 0 - the evidence
    ax = fig.add_subplot(gs[0, 0]); show(ax, img, "MRI + evidence region (red)\nbrain outline (green)")
    ax.contour(A["hotspot"].astype(float), levels=[0.5], colors="red", linewidths=1.8)
    ax.contour(A["brain"].astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    ttl = f"Pred: {names[r['pred']]} ({r['confidence']:.0%})"
    if r["true_label"] is not None:
        ttl += f" | True: {names[r['true_label']]}"
    ax.set_xlabel(ttl, fontsize=10)
    ax.xaxis.set_visible(True); ax.set_xticks([])
    show(fig.add_subplot(gs[0, 1]), overlay(img, A["consensus"]),
         f"Consensus of {len(r['consensus']['methods_used'])} methods\nfaithfulness {r['consensus']['faithfulness']:+.2f}")
    prim = r["consensus"]["primary"]
    show(fig.add_subplot(gs[0, 2]), overlay(img, A["maps"][prim]),
         f"Best single method: {r['methods'][prim]['label']}\nfaithfulness {r['methods'][prim]['faithfulness']:+.2f}")
    show(fig.add_subplot(gs[0, 3]), A["removed"],
         f"Region REMOVED\nP({names[c]}) {cz['p_orig']:.2f} -> {cz['p_removed']:.2f}")
    show(fig.add_subplot(gs[0, 4]), A["kept"],
         f"ONLY region kept\nP({names[c]}) = {cz['p_kept']:.2f} ({cz['sufficiency']:.0%})")
    if A.get("attention") is not None:
        show(fig.add_subplot(gs[0, 5]), overlay(img, A["attention"], "hot"),
             f"Built-in attention gate\nrho vs consensus {r['attention']['spearman_with_consensus']:+.2f}")
    elif A.get("contrast_map") is not None:
        show(fig.add_subplot(gs[0, 5]), overlay(img, A["contrast_map"]),
             f"Evidence for runner-up '{names[r['second']]}'")
    else:
        fig.add_subplot(gs[0, 5]).axis("off")

    # row 1 - every method
    for i, (name, m) in enumerate(list(A["maps"].items())[:6]):
        show(fig.add_subplot(gs[1, i]), overlay(img, m),
             f"{r['methods'][name]['label']}\nins {r['methods'][name]['auc_insertion']:.2f} / del "
             f"{r['methods'][name]['auc_deletion']:.2f}", fs=9)

    # row 2 - curves, probabilities, controls
    ax = fig.add_subplot(gs[2, 0:2])
    cv = r["consensus"]["curves"]
    ax.plot(cv["fractions"], cv["insertion"], "g-o", ms=3, label=f"insertion (AUC {r['consensus']['auc_insertion']:.2f})")
    ax.plot(cv["fractions"], cv["deletion"], "r-o", ms=3, label=f"deletion (AUC {r['consensus']['auc_deletion']:.2f})")
    ax.set_xlabel("fraction of pixels revealed / removed (most important first)")
    ax.set_ylabel(f"P({names[c]})"); ax.set_ylim(-0.02, 1.02); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Faithfulness curves (consensus map)", fontsize=10)

    ax = fig.add_subplot(gs[2, 2:4])
    x = np.arange(len(names)); wd = 0.27
    ax.bar(x - wd, r["probs_calibrated"], wd, label="original")
    ax.bar(x, cz["probs_removed"], wd, label="region removed")
    ax.bar(x + wd, cz["probs_kept"], wd, label="only region kept")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("Class probabilities under the counterfactual edits", fontsize=10)

    ax = fig.add_subplot(gs[2, 4:6])
    if cz["n_controls"]:
        ax.hist(cz["control_drops"], bins=12, color="grey", alpha=0.8, label="random same-shape regions")
    ax.axvline(cz["drop"], color="red", lw=2.5, label=f"evidence region (p = {cz['p_value']:.3f})")
    ax.set_xlabel(f"drop in P({names[c]}) after removal"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    ax.set_title("Is the region special? (specificity test)", fontsize=10)

    # row 3 - checklist + features
    ax = fig.add_subplot(gs[3, 0:3]); ax.axis("off")
    ev = r["evidence_strength"]
    lines = [f"EVIDENCE CHECKLIST  ->  {ev['label'].upper()}  ({ev['passed']}/{ev['total']})", ""]
    for ck in r["checklist"]:
        mark = "✔" if ck["passed"] else "✘"
        lines.append(f"{mark} {ck['name']}: {ck['value']}   [{ck['rule']}]")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9, family="DejaVu Sans")

    ax = fig.add_subplot(gs[3, 3:6]); ax.axis("off")
    f = r["features"]
    rows = ["FEATURES OF THE EVIDENCE REGION", ""]
    for k in ("area_frac", "dist_center", "depth", "contrast_z", "heterogeneity", "solidity", "compactness",
              "boundary_sharpness", "asymmetry_ratio", "in_brain_mass", "rim_mass"):
        rows.append(f"{k:<20}{f[k]:>8.2f}   {FEATURE_DESCRIPTIONS[k]}")
    sg = r.get("surrogate")
    if sg and sg["supporting"]:
        rows += ["", "Surrogate: features supporting the prediction"]
        rows += [f"  + {x['feature']:<18}{x['contribution']:+.2f}" for x in sg["supporting"]]
    ax.text(0, 1, "\n".join(rows), va="top", ha="left", fontsize=8.5, family="DejaVu Sans Mono")

    # row 4 - narrative
    ax = fig.add_subplot(gs[4, :]); ax.axis("off")
    wrapped = "\n\n".join(textwrap.fill(p, 205) for p in r["text"].split("\n\n"))
    ax.text(0, 1, wrapped, va="top", ha="left", fontsize=8.8, family="DejaVu Sans")
    fig.suptitle("Evidence-based explanation", fontsize=16, y=0.985)
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


# ======================================================================================
# Saving
# ======================================================================================
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if k != "arrays"}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def to_json_safe(result: dict) -> dict:
    return _clean(result)


def save_report(result: dict, stem: str, figure: bool = True, dpi: int = 110) -> dict:
    """Write ``<stem>.json`` (all numbers), ``<stem>.md`` (text + checklist) and ``<stem>.png``."""
    import matplotlib.pyplot as plt
    paths = {"json": stem + ".json", "md": stem + ".md"}
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(to_json_safe(result), fh, indent=2)
    md = ["# Evidence report", "", result["text"].replace("\n\n", "\n\n"), "", "## Checklist", "",
          "| | check | value | rule |", "|---|---|---|---|"]
    for ck in result["checklist"]:
        md.append(f"| {'PASS' if ck['passed'] else 'FAIL'} | {ck['name']} | {ck['value']} | {ck['rule']} |")
    with open(paths["md"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    if figure and "arrays" in result:
        paths["png"] = stem + ".png"
        fig = make_evidence_figure(result, paths["png"], dpi=dpi)
        plt.close(fig)
    return paths
