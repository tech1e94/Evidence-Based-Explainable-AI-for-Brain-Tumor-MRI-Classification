"""xai_brain - evidence-based explainability toolkit for the brain-tumour MRI classifier.

Design note: this package never imports TensorFlow at import time. Only
``model``, ``saliency`` (gradient part), ``channels`` and ``data.make_tf_dataset``
need TensorFlow, and they import it lazily. Everything else (evidence extraction,
faithfulness metrics, calibration, surrogate model, report generation) works on
plain NumPy arrays plus a ``predict_fn(images) -> probabilities`` callable.
"""

from .config import CATEGORIES, IMG_SIZE, EvidenceConfig  # noqa: F401

__version__ = "1.0.0"
