"""Central configuration: class names, image size and the evidence-checklist thresholds."""
from dataclasses import dataclass, field

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224


@dataclass
class EvidenceConfig:
    """All tunable numbers used by the evidence report in one place.

    The thresholds below are *conventions*, not clinical standards. They are the
    pass/fail lines of the transparent checklist in the report. Tune them on your
    validation set (``run_xai_evaluation.py`` prints the distribution of every
    metric so you can pick sensible values).
    """

    # --- hotspot extraction -------------------------------------------------
    hotspot_top_frac: float = 0.10      # hotspot = top 10 % of image pixels of the consensus map
    hotspot_min_area: int = 30          # ignore specks smaller than this (pixels)
    hotspot_max_components: int = 3
    consensus_smooth_sigma: float = 3.0

    # --- faithfulness curves --------------------------------------------------
    curve_steps: int = 20
    fill_mode: str = "mean"             # 'mean' (brain tissue) | 'blur' | 'inpaint' (removal tests)

    # --- counterfactual (necessity / sufficiency) tests --------------------------
    n_random_controls: int = 30
    random_seed: int = 0

    # --- uncertainty ---------------------------------------------------------------
    n_tta: int = 16
    n_stability: int = 5
    stability_sigma: float = 0.02

    # --- checklist thresholds ----------------------------------------------------------
    min_faithfulness: float = 0.05      # insertion AUC - deletion AUC of the consensus map
    min_necessity_drop: float = 0.15    # absolute fall in P(class) when hotspot is removed
    max_control_p: float = 0.10         # empirical p-value vs random same-size regions
    min_sufficiency: float = 0.50       # P(class | hotspot only) / P(class | full image)
    min_agreement: float = 0.35         # mean pairwise Spearman between methods
    min_in_brain_mass: float = 0.90     # share of explanation mass inside the brain
    max_rim_mass: float = 0.40          # share on the outer rim (skull-edge artefact check)
    min_stability: float = 0.60         # Spearman between map(img) and map(img + noise)
    min_confidence: float = 0.60        # calibrated top-1 probability
    min_margin: float = 0.20            # top-1 minus top-2 probability
    min_tta_agreement: float = 0.80     # share of augmentations giving the same class

    strong_cut: float = 0.80            # share of applicable checks passed -> "Strong"
    moderate_cut: float = 0.50          # -> "Moderate", below -> "Weak"
