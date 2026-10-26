# Evidence report

PREDICTION - notumor. Calibrated confidence 100% (raw 100%), margin 0.99 over 'meningioma' (0%). Under 16 small test-time changes the same class was returned 100% of the time. Ground truth: notumor -> the prediction is CORRECT.

WHERE - Combining 4 explanation method(s); the methods agree with Spearman rho = 0.46, the strongest evidence for 'notumor' forms 3 region(s) covering 3.2% of the brain (small), located middle, in the right half of the image, at mid-depth. 93% of the explanation lies inside the brain and 16% on its outer rim. Best single method by faithfulness: Integrated Gradients (insertion-deletion = +0.40).

WHICH FEATURES - Inside this region the tissue is similar in brightness to the surrounding brain (-0.4 SD); the region is elongated with a sharp boundary, heterogeneous texture (heterogeneity 1.02, entropy 2.68 bits) and markedly asymmetric vs. the opposite side. Compared with correctly classified 'notumor' cases in your data, the most typical features are boundary_sharpness (z=+0.1), glcm_contrast (z=-0.2), compactness (z=-0.3); unusual: midline_offset (z=+2.5).

FEATURE ATTRIBUTION - An interpretable surrogate (agrees with the CNN 75% of the time vs 30% for always guessing the majority class) credits these features for the prediction: heterogeneity=1.02 (+0.63), compactness=0.11 (+0.40), contrast_z=-0.39 (+0.37), rel_y=0.10 (+0.31); against it: midline_offset=0.65 (-0.26), eccentricity=0.97 (-0.25), n_components=3.00 (-0.18), glcm_homogeneity=0.68 (-0.09).

CAUSAL CHECK - Removing that region lowers P(notumor) from 1.00 to 1.00 (-0.00) but the predicted class stays the same; random same-shape regions lower it by only 0.00 on average (empirical p = 0.323, n = 30). Keeping only the region retains 0% of the original probability.

INTERNAL FEATURES - Feature channels [158, 89, 126, 45, 188] of the last convolution contribute most to the class score. Switching them off lowers P by 0.13 versus 0.00 for the same number of random channels.

WHY NOT 'meningioma'? - Its probability is 0%; after the evidence region is removed it becomes 0%. The evidence map for 'meningioma' correlates +0.86 with the one for 'notumor' (similar regions are used).

SIMILAR CASES - Closest reference images in the model's feature space: notumor (0.97), notumor (0.96), notumor (0.94), notumor (0.94), notumor (0.92). 100% share the predicted class.

EVIDENCE STRENGTH - MODERATE (9/12 checks passed); not satisfied: Region is NECESSARY (removal drops P(class)); Region is SPECIFIC (vs random same-shape regions); Region is SUFFICIENT (keep only region).

CAVEATS - The highlighted region is where the MODEL looked, not a radiologist's tumour outline; intensity words describe the image, not a diagnosis; thresholds are heuristic conventions. This is research software, not a medical device.

## Checklist

| | check | value | rule |
|---|---|---|---|
| PASS | Calibrated confidence | 100% | >= 60% |
| PASS | Margin over runner-up | 0.99 | >= 0.20 |
| PASS | Prediction stable under augmentation | 100% | >= 80% |
| PASS | Explanation is faithful (insertion - deletion AUC) | +0.32 | >= +0.05 |
| FAIL | Region is NECESSARY (removal drops P(class)) | -0.00 | drop >= 0.15 or flip |
| FAIL | Region is SPECIFIC (vs random same-shape regions) | p = 0.323 | p <= 0.1 |
| FAIL | Region is SUFFICIENT (keep only region) | 0% of original | >= 50% |
| PASS | Explanation methods agree (mean Spearman) | 0.46 | >= 0.35 |
| PASS | Evidence lies inside the brain | 93% | >= 90% |
| PASS | Evidence is not just the skull edge | 16% on rim | <= 40% |
| PASS | Explanation stable under tiny noise | 0.98 | >= 0.60 |
| PASS | Explanation depends on learned weights (randomisation test) | |rho| = 0.15 | |rho| < 0.5 |