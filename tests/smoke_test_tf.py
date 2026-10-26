"""Run this FIRST on your machine (needs TensorFlow, no dataset):  python tests/smoke_test_tf.py

Builds both architectures with random weights and pushes a random image through EVERY
TensorFlow-based explainer, the channel-ablation test, the randomisation test and a full
evidence report. It checks shapes/ranges only (random weights explain nothing)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import numpy as np

from xai_brain.channels import channel_evidence
from xai_brain.config import CATEGORIES, EvidenceConfig
from xai_brain.model import (build_baseline_model, build_improved_model, embedding_model, make_mc_dropout_fn,
                             make_predict_fn)
from xai_brain.report import explain_image, make_evidence_figure
from xai_brain.saliency import build_explainers, parameter_randomization_test

rng = np.random.default_rng(0)
img = np.clip(rng.random((224, 224, 3)).astype(np.float32) * 0.3 + 0.1, 0, 1)
for name, build in (("baseline", build_baseline_model), ("improved", build_improved_model)):
    print(f"--- {name}")
    model = build()
    predict = make_predict_fn(model)
    ex, ge = build_explainers(model, predict, ("gradcam", "gradcampp", "scorecam", "ig", "smoothgrad", "occlusion"))
    c = int(predict(img[None])[0].argmax())
    for k, fn in ex.items():
        m = fn(img, c)
        assert m.shape == (224, 224) and 0 <= m.min() and m.max() <= 1.0 + 1e-6, (k, m.shape, m.min(), m.max())
        print("  ok", k)
    a = ge.attention(img); assert a.shape == (224, 224); print("  ok attention gate")
    w0 = [w.copy() for w in ge.conv.get_weights()]
    ch = channel_evidence(ge, img, c, predict)
    assert all(np.allclose(a_, b_) for a_, b_ in zip(w0, ge.conv.get_weights())), "weights not restored!"
    print("  ok channel evidence", ch["channels"], "ablation:", ch["ablation_supported"])
    s = parameter_randomization_test(model, [img], [c]); print("  ok randomisation", round(s["mean_abs_rho_full_random"], 2))
    print("  embedding dim", embedding_model(model).output_shape)
    res = explain_image(img, predict, {k: ex[k] for k in ("gradcam", "gradcampp", "ig", "occlusion")}, CATEGORIES,
                        config=EvidenceConfig(n_random_controls=6, n_tta=4, n_stability=2),
                        attention_fn=ge.attention, channel_fn=lambda im, k: channel_evidence(ge, im, k, predict),
                        mc_fn=make_mc_dropout_fn(model, 6) if name == "improved" else None)
    make_evidence_figure(res, os.path.join("outputs", f"_smoke_{name}.png"))
    print("  ok full report ->", res["evidence_strength"]["label"])
print("\nSMOKE TEST PASSED")
