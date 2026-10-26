# Brain Tumor Detection with Evidence-Based Explainable AI

MRI brain-tumour classification (glioma / meningioma / no tumour / pituitary) with a CNN + spatial attention gate,
extended into an **evidence-based XAI toolkit**: every explanation is backed by measurable features and causal tests,
not just a heat-map.

> Research / educational software - **not a medical device**.

## What is new
| Need | What the toolkit does | Where |
|---|---|---|
| "Which feature is responsible?" | Measures the region the model relied on: location, size, depth, brightness vs. surrounding brain (z-score), texture, shape, boundary sharpness, left/right asymmetry. Compares them with typical cases of the predicted class (z-scores) and credits them with an interpretable **surrogate model** (logistic regression + shallow tree, with reported fidelity). Also ranks the CNN's internal feature channels and switches them off to test them. | `evidence.py`, `surrogate.py`, `channels.py` |
| "Provide evidence" | **Necessity** (remove region -> P(class) falls), **sufficiency** (keep only region), **specificity** (vs. 30 random same-shape regions, empirical p-value), deletion/insertion **faithfulness**, **agreement** between 5 methods, **stability** under noise, evidence **inside the brain / not on skull rim**, **randomisation sanity check**. | `evidence.py`, `metrics.py`, `saliency.py` |
| "More explainable" | Methods: Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, SmoothGrad, Occlusion, LIME, built-in attention gate -> consensus map. **Contrastive** "why not class Y?", **nearest reference cases**, class-average maps, t-SNE, error analysis. Natural-language explanation whose every sentence comes from a stored number. | `report.py`, `global_analysis.py` |
| Trustworthy confidence | Temperature-scaling calibration, ECE + reliability diagram, test-time-augmentation and MC-dropout uncertainty. | `uncertainty.py` |
| Transparent trust verdict | An 11-point pass/fail **checklist** (thresholds in `EvidenceConfig`) replaces the single heuristic. | `report.py` |
| Better model | `train.py --arch improved`: BatchNorm + dropout + augmentation + same attention gate. | `model.py`, `train.py` |
| Demo | Streamlit app: upload an MRI, get the full evidence card. | `app.py` |

Example evidence card (**synthetic** image + stand-in model, only to show the layout): `docs/synthetic_example_card.png`

## Quick start
```bash
pip install -r requirements.txt
python tests/smoke_test_tf.py                       # 1 min: checks every TensorFlow explainer runs (no data needed)
python train.py --data archive --arch improved      # or save your notebook model: model.save("models/baseline_cnn.keras")
jupyter notebook tumor_xai_evidence.ipynb           # guided walkthrough
python run_xai_evaluation.py --model models/improved_cnn.keras --data archive   # dataset-level report -> outputs/
streamlit run app.py
```
`archive/` is the Kaggle brain-tumour MRI folder with `Training/` and `Testing/` (folder-name case does not matter).

```python
res = explain_image(img, predict_fn, explainers, CATEGORIES, true_label=y, temperature=T, profile=prof, surrogate=sur)
print(res["text"]); make_evidence_figure(res, "card.png"); save_report(res, "outputs/case1")
```

## How to read the evidence checklist
| Check | Passes when | Meaning |
|---|---|---|
| Calibrated confidence / margin / TTA stability | >=60% / >=0.20 / >=80% | the prediction itself is firm |
| Faithfulness | insertion-AUC - deletion-AUC >= 0.05 | the map predicts what the model reacts to |
| Necessary | removing the region drops P(class) by >=0.15 or flips it | the model needs it |
| Specific | p <= 0.10 vs random same-shape regions | not just any removal hurts |
| Sufficient | region alone keeps >=50% of P(class) | region carries the decision |
| Methods agree | mean Spearman >= 0.35 | independent methods point to the same place |
| Inside brain / not rim | >=90% / <=40% | not a background or skull-edge shortcut |
| Stable explanation | rho >= 0.6 under tiny noise | map is not fragile |
| Randomisation test | \|rho\| < 0.5 | map depends on learned weights |

Thresholds are conventions - tune them on your validation distribution (`outputs/xai_metrics.csv`).

## Issues found in the original notebook (fixed in the new code)
1. **"Attention Map" showed the wrong layer.** `get_layer(index=4)` is the 2nd *MaxPooling*, not the attention gate. The toolkit finds the sigmoid gate by type.
2. **The "trust" rule could never say HIGH**: `attn_ent < 5`, but the entropy of a 54x54 map is ~7.5 (max 7.98). The entropy of a min-max-normalised map is also not a trust measure. Replaced by the checklist.
3. **"Tumor Region (Circled)"** draws a hard-coded circle at (110, 80) on the first test image - it is not a detection. Don't use it in a report.
4. **Every explanation demo used one image** (`test_generator.reset(); next()`) - a *misclassified glioma*. Conclusions need dataset-level statistics (`run_xai_evaluation.py`).
5. `load_data` reads with OpenCV (**BGR**) while the model was trained via the generator (**RGB**); `"archive/testing"` vs `"Testing"` breaks on Linux/Colab; loading all 5,600 images as float64 needs ~6.7 GB RAM. `xai_brain.data` handles all three.
6. The baseline reaches ~70% test accuracy (glioma recall 49%), so its explanations are weak by construction - the checklist will (correctly) often say "Weak/Moderate". Train the improved model first.

## Structure
```
xai_brain/        evidence.py metrics.py uncertainty.py surrogate.py saliency.py channels.py
                  report.py evaluate.py global_analysis.py model.py data.py utils.py config.py
tests/            test_core_numpy.py (no TF needed)  smoke_test_tf.py (needs TF)  synthetic.py
train.py  run_xai_evaluation.py  app.py  tumor_enhanced.ipynb (original)  tumor_xai_evidence.ipynb (new)
```

## Testing status (be aware)
* `tests/test_core_numpy.py` passes: evidence, faithfulness, counterfactual tests, calibration, surrogate, report/figure,
  dataset-level evaluation - on **synthetic** MRI-like images with a stand-in classifier.
* The TensorFlow parts (Grad-CAM/++/Score-CAM/IG/SmoothGrad, channel ablation, MC-dropout, randomisation test, `train.py`,
  `app.py`) were written without a TensorFlow runtime available and **have not been executed**. Run
  `python tests/smoke_test_tf.py` first and report any error.
* Nothing has been run on the real MRI dataset; the thresholds and any accuracy figure for the improved model are untested.

## Limitations
The highlighted region is where the *model* looked, not a tumour segmentation; intensity descriptions are image statistics, not diagnoses;
slice orientation varies in this dataset, so anatomical words are deliberately generic; the Kaggle data has no patient-level split guarantees, so test accuracy may be optimistic.

## Author
Gaurav
