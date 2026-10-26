"""Streamlit demo:  streamlit run app.py   (needs a saved model in models/)"""
import glob
import json
import os

import joblib
import matplotlib
import numpy as np
import streamlit as st
from PIL import Image

matplotlib.use("Agg")
from xai_brain.config import CATEGORIES, EvidenceConfig
from xai_brain.report import explain_image, make_evidence_figure
from xai_brain.utils import to_float01

st.set_page_config(page_title="Evidence-based MRI explanations", layout="wide")
st.title("Brain-tumour MRI: evidence-based explanation")
st.caption("Research demo - not a medical device.")


@st.cache_resource
def load_everything(model_path):
    from xai_brain.channels import channel_evidence
    from xai_brain.model import load_model, make_predict_fn
    from xai_brain.saliency import build_explainers
    model = load_model(model_path)
    predict = make_predict_fn(model)
    explainers, ge = build_explainers(model, predict, ("gradcam", "gradcampp", "scorecam", "ig", "occlusion"))
    prof = json.load(open("outputs/class_profiles.json")) if os.path.exists("outputs/class_profiles.json") else None
    prof = {k: {f: tuple(v) for f, v in d.items()} for k, d in prof.items()} if prof else None
    sur = joblib.load("outputs/surrogate.joblib") if os.path.exists("outputs/surrogate.joblib") else None
    return predict, explainers, ge, prof, sur, channel_evidence


models = sorted(glob.glob("models/*.keras"))
if not models:
    st.error("No model found in models/. Run train.py or save your notebook model there.")
    st.stop()
path = st.sidebar.selectbox("Model", models)
temperature = st.sidebar.number_input("Calibration temperature", 0.1, 10.0, 1.0, 0.05)
n_ctrl = st.sidebar.slider("Random control regions", 10, 60, 30)
file = st.file_uploader("Upload a brain MRI slice (jpg/png)", type=["jpg", "jpeg", "png"])
if file:
    predict, explainers, ge, prof, sur, channel_evidence = load_everything(path)
    img = to_float01(np.asarray(Image.open(file).convert("RGB").resize((224, 224))))
    with st.spinner("Computing explanations..."):
        res = explain_image(img, predict, explainers, CATEGORIES, config=EvidenceConfig(n_random_controls=n_ctrl),
                            profile=prof, surrogate=sur, temperature=temperature,
                            attention_fn=ge.attention if ge.gate_model else None,
                            channel_fn=lambda im, k: channel_evidence(ge, im, k, predict))
    st.pyplot(make_evidence_figure(res), use_container_width=True)
    st.subheader("Explanation"); st.text(res["text"])
    st.subheader("Checklist")
    st.table([{"": "PASS" if c["passed"] else "FAIL", "check": c["name"], "value": c["value"], "rule": c["rule"]}
              for c in res["checklist"]])
