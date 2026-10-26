# Evidence report

PREDICTION - meningioma. Calibrated confidence 63% (raw 68%), margin 0.28 over 'glioma' (35%). Under 16 small test-time changes the same class was returned 75% of the time. Ground truth: glioma -> the prediction is WRONG.

WHERE - Combining 3 explanation method(s); the methods agree with Spearman rho = 0.47, the strongest evidence for 'meningioma' forms 2 region(s) covering 4.7% of the brain (small), located upper, near the midline, deep inside the brain. 98% of the explanation lies inside the brain and 3% on its outer rim. Best single method by faithfulness: Grad-CAM (insertion-deletion = +0.61).

WHICH FEATURES - Inside this region the tissue is similar in brightness to the surrounding brain (+0.0 SD); the region is round and compact with a sharp boundary, heterogeneous texture (heterogeneity 0.72, entropy 2.67 bits) and mildly asymmetric vs. the opposite side. Compared with correctly classified 'meningioma' cases in your data, the most typical features are rel_y (z=-0.1), area_frac (z=-0.6), midline_offset (z=-0.6); unusual: compactness (z=+4.5).

FEATURE ATTRIBUTION - An interpretable surrogate (agrees with the CNN 75% of the time vs 30% for always guessing the majority class) credits these features for the prediction: compactness=0.90 (+1.77), rim_mass=0.03 (+0.39), depth=0.78 (+0.27), eccentricity=0.74 (+0.27); against it: solidity=0.95 (-0.75), in_brain_mass=0.98 (-0.60), asymmetry_ratio=1.34 (-0.22), n_components=2.00 (-0.15).

CAUSAL CHECK - Removing that region lowers P(meningioma) from 0.68 to 0.51 (-0.16) but the predicted class stays the same; random same-shape regions lower it by only -0.24 on average (empirical p = 0.032, n = 30). Keeping only the region retains 144% of the original probability.

INTERNAL FEATURES - Feature channels [166, 7, 124, 76, 185] of the last convolution contribute most to the class score. Switching them off lowers P by 0.37 versus 0.01 for the same number of random channels.

WHY NOT 'glioma'? - Its probability is 35%; after the evidence region is removed it becomes 38%. The evidence map for 'glioma' correlates -0.28 with the one for 'meningioma' (different regions are used).

SIMILAR CASES - Closest reference images in the model's feature space: glioma (0.98), meningioma (0.97), glioma (0.97), meningioma (0.97), meningioma (0.96). 60% share the predicted class.

EVIDENCE STRENGTH - STRONG (11/12 checks passed); not satisfied: Prediction stable under augmentation.

CAVEATS - The highlighted region is where the MODEL looked, not a radiologist's tumour outline; intensity words describe the image, not a diagnosis; thresholds are heuristic conventions. This is research software, not a medical device.

## Checklist

| | check | value | rule |
|---|---|---|---|
| PASS | Calibrated confidence | 63% | >= 60% |
| PASS | Margin over runner-up | 0.28 | >= 0.20 |
| FAIL | Prediction stable under augmentation | 75% | >= 80% |
| PASS | Explanation is faithful (insertion - deletion AUC) | +0.67 | >= +0.05 |
| PASS | Region is NECESSARY (removal drops P(class)) | -0.16 | drop >= 0.15 or flip |
| PASS | Region is SPECIFIC (vs random same-shape regions) | p = 0.032 | p <= 0.1 |
| PASS | Region is SUFFICIENT (keep only region) | 144% of original | >= 50% |
| PASS | Explanation methods agree (mean Spearman) | 0.47 | >= 0.35 |
| PASS | Evidence lies inside the brain | 98% | >= 90% |
| PASS | Evidence is not just the skull edge | 3% on rim | <= 40% |
| PASS | Explanation stable under tiny noise | 0.92 | >= 0.60 |
| PASS | Explanation depends on learned weights (randomisation test) | |rho| = 0.15 | |rho| < 0.5 |