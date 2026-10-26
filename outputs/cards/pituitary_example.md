# Evidence report

PREDICTION - pituitary. Calibrated confidence 99% (raw 100%), margin 0.98 over 'meningioma' (1%). Under 16 small test-time changes the same class was returned 100% of the time. Ground truth: pituitary -> the prediction is CORRECT.

WHERE - Combining 4 explanation method(s); the methods agree with Spearman rho = 0.74, the strongest evidence for 'pituitary' forms 3 region(s) covering 7.2% of the brain (small), located lower, near the midline, deep inside the brain. 98% of the explanation lies inside the brain and 40% on its outer rim. Best single method by faithfulness: Occlusion (insertion-deletion = +0.49).

WHICH FEATURES - Inside this region the tissue is brighter than the surrounding brain (+1.5 SD); the region is elongated with a sharp boundary, heterogeneous texture (heterogeneity 0.43, entropy 2.99 bits) and mildly asymmetric vs. the opposite side. Compared with correctly classified 'pituitary' cases in your data, the most typical features are in_brain_mass (z=-0.1), solidity (z=-0.3), depth (z=+0.4).

FEATURE ATTRIBUTION - An interpretable surrogate (agrees with the CNN 75% of the time vs 30% for always guessing the majority class) credits these features for the prediction: rim_mass=0.40 (+0.56), in_brain_mass=0.98 (+0.43), dist_center=0.28 (+0.38), eccentricity=0.99 (+0.33); against it: midline_offset=0.04 (-0.31), texture_entropy=2.99 (-0.19), area_frac=0.07 (-0.17), glcm_homogeneity=0.61 (-0.11).

CAUSAL CHECK - Removing that region lowers P(pituitary) from 1.00 to 0.77 (-0.23) but the predicted class stays the same; random same-shape regions lower it by only 0.00 on average (empirical p = 0.042, n = 23). Keeping only the region retains 100% of the original probability.

INTERNAL FEATURES - Feature channels [78, 102, 93, 7, 28] of the last convolution contribute most to the class score. Switching them off lowers P by 0.82 versus 0.00 for the same number of random channels.

WHY NOT 'meningioma'? - Its probability is 1%; after the evidence region is removed it becomes 14%. The evidence map for 'meningioma' correlates -0.81 with the one for 'pituitary' (different regions are used).

SIMILAR CASES - Closest reference images in the model's feature space: pituitary (0.99), pituitary (0.99), pituitary (0.99), pituitary (0.99), pituitary (0.99). 100% share the predicted class.

EVIDENCE STRENGTH - STRONG (12/12 checks passed).

CAVEATS - The highlighted region is where the MODEL looked, not a radiologist's tumour outline; intensity words describe the image, not a diagnosis; thresholds are heuristic conventions. This is research software, not a medical device.

## Checklist

| | check | value | rule |
|---|---|---|---|
| PASS | Calibrated confidence | 99% | >= 60% |
| PASS | Margin over runner-up | 0.98 | >= 0.20 |
| PASS | Prediction stable under augmentation | 100% | >= 80% |
| PASS | Explanation is faithful (insertion - deletion AUC) | +0.36 | >= +0.05 |
| PASS | Region is NECESSARY (removal drops P(class)) | -0.23 | drop >= 0.15 or flip |
| PASS | Region is SPECIFIC (vs random same-shape regions) | p = 0.042 | p <= 0.1 |
| PASS | Region is SUFFICIENT (keep only region) | 100% of original | >= 50% |
| PASS | Explanation methods agree (mean Spearman) | 0.74 | >= 0.35 |
| PASS | Evidence lies inside the brain | 98% | >= 90% |
| PASS | Evidence is not just the skull edge | 40% on rim | <= 40% |
| PASS | Explanation stable under tiny noise | 0.98 | >= 0.60 |
| PASS | Explanation depends on learned weights (randomisation test) | |rho| = 0.15 | |rho| < 0.5 |