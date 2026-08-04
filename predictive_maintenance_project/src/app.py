"""Streamlit dashboard for the AI4I 2020 predictive-maintenance project."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model_definitions import LSTMClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


@st.cache_resource
def load_models():
    rf = joblib.load(MODEL_DIR / "random_forest_pipeline.joblib")
    svm = joblib.load(MODEL_DIR / "svm_pipeline.joblib")
    lstm_pre = joblib.load(MODEL_DIR / "lstm_preprocessor.joblib")
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = LSTMClassifier(input_dim=8, hidden_dim=16, dropout=0.2)
    state = torch.load(MODEL_DIR / "lstm_model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    thresholds = {
        row["Model"]: row["Decision threshold"] for row in metadata["metrics"]
    }
    return rf, svm, lstm_pre, model, metadata, thresholds


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def validate_input(data: pd.DataFrame) -> tuple[bool, str]:
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        return False, "Missing required columns: " + ", ".join(missing)
    if not data["Type"].astype(str).isin(["L", "M", "H"]).all():
        return False, "Type must contain only L, M or H."
    numeric_columns = [column for column in FEATURES if column != "Type"]
    converted = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        return False, "One or more numerical feature columns contain invalid values."
    return True, "Input data passed validation."


def predict_all(data: pd.DataFrame, rf, svm, lstm_pre, lstm, thresholds):
    features = data[FEATURES].copy()
    rf_prob = rf.predict_proba(features)[:, 1]
    svm_prob = svm.predict_proba(features)[:, 1]
    transformed = lstm_pre.transform(features).astype("float32")[:, None, :]
    with torch.no_grad():
        lstm_prob = torch.sigmoid(lstm(torch.from_numpy(transformed))).numpy()
    output = pd.DataFrame(
        {
            "Random Forest probability": rf_prob,
            "Random Forest prediction": (rf_prob >= thresholds["Random Forest"]).astype(int),
            "SVM probability": svm_prob,
            "SVM prediction": (svm_prob >= thresholds["SVM"]).astype(int),
            "LSTM probability": lstm_prob,
            "LSTM prediction": (lstm_prob >= thresholds["LSTM"]).astype(int),
        }
    )
    return output


st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("AI4I 2020 Predictive Maintenance Dashboard")
st.caption(
    "Comparison of Random Forest, Support Vector Machine and an exploratory one-step LSTM."
)

rf, svm, lstm_pre, lstm, metadata, thresholds = load_models()
default_data = load_default_data()
metrics = pd.read_csv(TABLE_DIR / "model_metrics.csv")
efficiency = pd.read_csv(TABLE_DIR / "computational_efficiency.csv")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data", ["Default AI4I dataset", "Upload compatible CSV"])
    uploaded = None
    if source == "Upload compatible CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.info(
        "Required features: Type, air temperature, process temperature, rotational speed, torque and tool wear."
    )

if source == "Default AI4I dataset":
    active_data = default_data
elif uploaded is not None:
    try:
        active_data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()
else:
    st.warning("Upload a CSV file to continue.")
    st.stop()

valid, message = validate_input(active_data)
if not valid:
    st.error(message)
    st.stop()
st.success(message)

pages = st.tabs(
    ["Overview", "Dataset", "Predictions", "Model comparison", "Explainability", "Efficiency", "Limitations"]
)

with pages[0]:
    st.subheader("Project overview")
    st.write(
        "The artefact compares predictive performance, computational efficiency and model explainability under a common AI4I 2020 benchmark."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset rows", f"{len(default_data):,}")
    col2.metric("Observed failures", int(default_data["Machine failure"].sum()))
    col3.metric("Models", 3)
    st.image(str(FIGURE_DIR / "figure_1_system_architecture.png"), use_container_width=True)

with pages[1]:
    st.subheader("Dataset inspection")
    st.write(f"Rows: **{len(active_data):,}** | Columns: **{active_data.shape[1]}**")
    st.dataframe(active_data.head(100), use_container_width=True)
    if "Machine failure" in active_data.columns:
        st.bar_chart(active_data["Machine failure"].value_counts().sort_index())
    st.write("Missing values by column")
    st.dataframe(active_data.isna().sum().rename("Missing values"), use_container_width=True)

with pages[2]:
    st.subheader("Generate predictions")
    max_rows = min(len(active_data), 500)
    rows = st.slider("Number of rows to score", 1, max_rows, min(50, max_rows))
    scored = active_data.iloc[:rows].copy()
    predictions = predict_all(scored, rf, svm, lstm_pre, lstm, thresholds)
    displayed = pd.concat([scored.reset_index(drop=True), predictions], axis=1)
    st.dataframe(displayed, use_container_width=True)
    st.download_button(
        "Download predictions",
        data=displayed.to_csv(index=False).encode("utf-8"),
        file_name="predictive_maintenance_predictions.csv",
        mime="text/csv",
    )

with pages[3]:
    st.subheader("Predictive performance")
    st.dataframe(metrics.style.format(precision=4), use_container_width=True)
    st.image(str(FIGURE_DIR / "model_performance_comparison.png"), use_container_width=True)
    st.image(str(FIGURE_DIR / "roc_curves.png"), use_container_width=True)
    selected_model = st.selectbox("Confusion matrix", ["Random Forest", "SVM", "LSTM"])
    matrix_file = {
        "Random Forest": "confusion_matrix_random_forest.png",
        "SVM": "confusion_matrix_svm.png",
        "LSTM": "confusion_matrix_lstm.png",
    }[selected_model]
    st.image(str(FIGURE_DIR / matrix_file), width=650)

with pages[4]:
    st.subheader("Feature importance and SHAP explanations")
    explanation_model = st.selectbox("Explanation model", ["Random Forest", "SVM", "LSTM"])
    if explanation_model == "Random Forest":
        st.image(str(FIGURE_DIR / "random_forest_feature_importance.png"), use_container_width=True)
        st.image(str(FIGURE_DIR / "shap_summary_random_forest.png"), use_container_width=True)
        st.image(str(FIGURE_DIR / "shap_local_random_forest.png"), use_container_width=True)
    elif explanation_model == "SVM":
        st.image(str(FIGURE_DIR / "shap_summary_svm.png"), use_container_width=True)
    else:
        st.image(str(FIGURE_DIR / "shap_summary_lstm.png"), use_container_width=True)
    st.warning(
        "SHAP values explain the fitted model's behaviour. They do not establish that a feature causes machine failure."
    )

with pages[5]:
    st.subheader("Computational efficiency")
    st.dataframe(efficiency.style.format(precision=4), use_container_width=True)
    st.image(str(FIGURE_DIR / "training_time.png"), use_container_width=True)
    st.image(str(FIGURE_DIR / "inference_time.png"), use_container_width=True)
    st.image(str(FIGURE_DIR / "peak_memory.png"), use_container_width=True)

with pages[6]:
    st.subheader("Important limitations")
    st.markdown(
        """
        - AI4I is a synthetic benchmark and does not reproduce all noise, drift and organisational constraints of live industrial systems.
        - The observations are not genuine machine-specific temporal histories. The LSTM is therefore a one-step exploratory comparator, not a full degradation-sequence model.
        - The dashboard is a research prototype and must not be used as a safety-critical industrial control system.
        - Model explanations describe fitted associations rather than causal mechanisms.
        """
    )
