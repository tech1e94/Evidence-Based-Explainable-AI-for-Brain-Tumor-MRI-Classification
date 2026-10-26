# Evidence report

PREDICTION - pituitary. Calibrated confidence 73% (raw 79%), margin 0.47 over 'meningioma' (26%). Under 16 small test-time changes the same class was returned 44% of the time. Ground truth: meningioma -> the prediction is WRONG.

WHERE - Combining 3 explanation method(s); the methods agree with Spearman rho = 0.72, the strongest evidence for 'pituitary' forms 2 region(s) covering 4.0% of the brain (small), located middle, near the midline, deep inside the brain. 100% of the explanation lies inside the brain and 12% on its outer rim. Best single method by faithfulness: Occlusion (insertion-deletion = +0.56).

WHICH FEATURES - Inside this region the tissue is brighter than the surrounding brain (+1.1 SD); the region is elongated with a sharp boundary, heterogeneous texture (heterogeneity 0.49, entropy 2.98 bits) and markedly asymmetric vs. the opposite side. Compared with correctly classified 'pituitary' cases in your data, the most typical features are boundary_sharpness (z=+0.1), eccentricity (z=-0.2), area_frac (z=-0.2); unusual: dist_center (z=-2.7), depth (z=+2.3), asymmetry_ratio (z=+2.0).

FEATURE ATTRIBUTION - An interpretable surrogate (agrees with the CNN 75% of the time vs 30% for always guessing the majority class) credits these features for the prediction: dist_center=0.08 (+0.68), solidity=0.84 (+0.62), in_brain_mass=1.00 (+0.52), depth=0.91 (+0.38); against it: rim_mass=0.12 (-0.28), midline_offset=0.08 (-0.23), texture_entropy=2.98 (-0.18), n_components=2.00 (-0.11).

CAUSAL CHECK - Removing that region lowers P(pituitary) from 0.79 to 0.21 (-0.58) and the prediction changes to 'meningioma'; random same-shape regions lower it by only 0.04 on average (empirical p = 0.032, n = 30). Keeping only the region retains 126% of the original probability.

INTERNAL FEATURES - Feature channels [78, 7, 102, 28, 160] of the last convolution contribute most to the class score. Switching them off lowers P by 0.72 versus -0.00 for the same number of random channels.

WHY NOT 'meningioma'? - Its probability is 26%; after the evidence region is removed it becomes 75%. The evidence map for 'meningioma' correlates -0.88 with the one for 'pituitary' (different regions are used).

SIMILAR CASES - Closest reference images in the model's feature space: pituitary (0.94), pituitary (0.94), pituitary (0.93), pituitary (0.93), pituitary (0.92). 100% share the predicted class.

EVIDENCE STRENGTH - STRONG (11/12 checks passed); not satisfied: Prediction stable under augmentation.

CAVEATS - The highlighted region is where the MODEL looked, not a radiologist's tumour outline; intensity words describe the image, not a diagnosis; thresholds are heuristic conventions. This is research software, not a medical device.

## Checklist

| | check | value | rule |
|---|---|---|---|
| PASS | Calibrated confidence | 73% | >= 60% |
| PASS | Margin over runner-up | 0.47 | >= 0.20 |
| FAIL | Prediction stable under augmentation | 44% | >= 80% |
| PASS | Explanation is faithful (insertion - deletion AUC) | +0.37 | >= +0.05 |
| PASS | Region is NECESSARY (removal drops P(class)) | -0.58 & class flips | drop >= 0.15 or flip |
| PASS | Region is SPECIFIC (vs random same-shape regions) | p = 0.032 | p <= 0.1 |
| PASS | Region is SUFFICIENT (keep only region) | 126% of original | >= 50% |
| PASS | Explanation methods agree (mean Spearman) | 0.72 | >= 0.35 |
| PASS | Evidence lies inside the brain | 100% | >= 90% |
| PASS | Evidence is not just the skull edge | 12% on rim | <= 40% |
| PASS | Explanation stable under tiny noise | 0.97 | >= 0.60 |
| PASS | Explanation depends on learned weights (randomisation test) | |rho| = 0.15 | |rho| < 0.5 |