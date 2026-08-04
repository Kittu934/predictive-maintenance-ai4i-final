"""Reproducible training pipeline for the AI4I 2020 predictive-maintenance study.

Run from the project root with:
    python src/train_pipeline.py

The script trains Random Forest, SVM and a one-step LSTM, evaluates predictive
performance and computational efficiency, saves models, and produces figures.
"""
from __future__ import annotations

import json
import os
import platform
import random
import threading
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_definitions import LSTMClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
for directory in [MODEL_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

TARGET = "Machine failure"
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
EXCLUDED = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CAT = ["Type"]
NUM = [column for column in FEATURES if column not in CAT]


class RSSMonitor:
    """Sample process RSS and report the maximum increase during a code block."""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.peak = 0

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.delta_mb = max(0, self.peak - self.baseline) / 1024**2


def best_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1[:-1]))
    return float(thresholds[index])


def metric_record(name: str, y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    prediction = (scores >= threshold).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1-score": f1_score(y_true, prediction, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, scores),
        "Decision threshold": threshold,
        "prediction": prediction,
    }


def train_lstm(
    x_train: np.ndarray,
    y_train: pd.Series,
    input_dim: int,
    hidden_units: int = 16,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 60,
) -> tuple[LSTMClassifier, float, float]:
    torch.manual_seed(SEED)
    model = LSTMClassifier(input_dim, hidden_units, dropout)
    positive_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train.to_numpy(np.float32))
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with RSSMonitor() as monitor:
        start = time.perf_counter()
        for _ in range(epochs):
            model.train()
            for features, labels in loader:
                optimiser.zero_grad()
                loss = loss_function(model(features), labels)
                loss.backward()
                optimiser.step()
        elapsed = time.perf_counter() - start
    model.eval()
    return model, elapsed, monitor.delta_mb


def clean_feature_names(names) -> list[str]:
    return [str(name).replace("cat__", "").replace("num__", "") for name in names]


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if data.empty:
        raise ValueError("Dataset is empty.")
    missing_columns = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    audit = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "machine_failure_counts": {
            str(key): int(value)
            for key, value in data[TARGET].value_counts().sort_index().items()
        },
    }
    (TABLE_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    x = data[FEATURES].copy()
    y = data[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    x_subtrain, x_validation, y_subtrain, y_validation = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    rf_preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("num", "passthrough", NUM),
        ]
    )
    rf_pipeline = Pipeline(
        [
            ("pre", rf_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced", random_state=SEED, n_jobs=-1
                ),
            ),
        ]
    )
    rf_search = GridSearchCV(
        rf_pipeline,
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt"],
        },
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )
    rf_search.fit(x_train, y_train)
    rf_validation = rf_search.best_estimator_.set_params()
    rf_validation.fit(x_subtrain, y_subtrain)
    rf_threshold = best_threshold(
        y_validation, rf_validation.predict_proba(x_validation)[:, 1]
    )

    svm_preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    svm_pipeline = Pipeline(
        [
            ("pre", svm_preprocessor),
            (
                "model",
                SVC(
                    class_weight="balanced",
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )
    svm_search = GridSearchCV(
        svm_pipeline,
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1, 1],
        },
        scoring="f1",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
    )
    svm_search.fit(x_train, y_train)
    svm_validation = svm_search.best_estimator_.set_params()
    svm_validation.fit(x_subtrain, y_subtrain)
    svm_threshold = best_threshold(
        y_validation, svm_validation.predict_proba(x_validation)[:, 1]
    )

    lstm_preprocessor_validation = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    lstm_subtrain = lstm_preprocessor_validation.fit_transform(x_subtrain).astype(np.float32)[:, None, :]
    lstm_validation = lstm_preprocessor_validation.transform(x_validation).astype(np.float32)[:, None, :]
    lstm_validation_model, _, _ = train_lstm(
        lstm_subtrain, y_subtrain, input_dim=lstm_subtrain.shape[2]
    )
    with torch.no_grad():
        validation_scores = torch.sigmoid(
            lstm_validation_model(torch.from_numpy(lstm_validation))
        ).numpy()
    lstm_threshold = best_threshold(y_validation, validation_scores)

    # Final models and timing
    rf_final = rf_search.best_estimator_
    with RSSMonitor() as rf_memory:
        start = time.perf_counter()
        rf_final.fit(x_train, y_train)
        rf_training_time = time.perf_counter() - start

    svm_final = svm_search.best_estimator_
    with RSSMonitor() as svm_memory:
        start = time.perf_counter()
        svm_final.fit(x_train, y_train)
        svm_training_time = time.perf_counter() - start

    lstm_preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    lstm_train = lstm_preprocessor.fit_transform(x_train).astype(np.float32)[:, None, :]
    lstm_test = lstm_preprocessor.transform(x_test).astype(np.float32)[:, None, :]
    lstm_final, lstm_training_time, lstm_memory = train_lstm(
        lstm_train, y_train, input_dim=lstm_train.shape[2]
    )

    def timed_scores(function, repeats: int = 5):
        durations = []
        output = None
        for _ in range(repeats):
            start = time.perf_counter()
            output = function()
            durations.append(time.perf_counter() - start)
        return output, float(np.median(durations))

    rf_scores, rf_inference = timed_scores(lambda: rf_final.predict_proba(x_test)[:, 1])
    svm_scores, svm_inference = timed_scores(lambda: svm_final.predict_proba(x_test)[:, 1])
    with torch.no_grad():
        lstm_scores, lstm_inference = timed_scores(
            lambda: torch.sigmoid(lstm_final(torch.from_numpy(lstm_test))).numpy()
        )

    records = [
        metric_record("Random Forest", y_test, rf_scores, rf_threshold),
        metric_record("SVM", y_test, svm_scores, svm_threshold),
        metric_record("LSTM", y_test, lstm_scores, lstm_threshold),
    ]
    metrics = pd.DataFrame(
        [{key: value for key, value in record.items() if key != "prediction"} for record in records]
    )
    metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)

    efficiency = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Training time (s)": rf_training_time,
                "Inference time for 2,000 cases (s)": rf_inference,
                "Peak memory increase (MB)": rf_memory.delta_mb,
            },
            {
                "Model": "SVM",
                "Training time (s)": svm_training_time,
                "Inference time for 2,000 cases (s)": svm_inference,
                "Peak memory increase (MB)": svm_memory.delta_mb,
            },
            {
                "Model": "LSTM",
                "Training time (s)": lstm_training_time,
                "Inference time for 2,000 cases (s)": lstm_inference,
                "Peak memory increase (MB)": lstm_memory,
            },
        ]
    )
    efficiency.to_csv(TABLE_DIR / "computational_efficiency.csv", index=False)

    joblib.dump(rf_final, MODEL_DIR / "random_forest_pipeline.joblib")
    joblib.dump(svm_final, MODEL_DIR / "svm_pipeline.joblib")
    joblib.dump(lstm_preprocessor, MODEL_DIR / "lstm_preprocessor.joblib")
    torch.save(lstm_final.state_dict(), MODEL_DIR / "lstm_model_state.pt")

    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "metrics": metrics.to_dict(orient="records"),
        "efficiency": efficiency.to_dict(orient="records"),
        "rf_best_parameters": rf_search.best_params_,
        "svm_best_parameters": svm_search.best_params_,
        "lstm_parameters": {
            "hidden_units": 16,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 60,
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Classification reports and confusion matrices
    for record in records:
        key = record["Model"].lower().replace(" ", "_")
        (REPORT_DIR / f"classification_report_{key}.txt").write_text(
            classification_report(y_test, record["prediction"], digits=4),
            encoding="utf-8",
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, record["prediction"]),
            display_labels=["No failure", "Failure"],
        ).plot(ax=axis)
        axis.set_title(f"Confusion Matrix: {record['Model']}")
        figure.tight_layout()
        figure.savefig(FIGURE_DIR / f"confusion_matrix_{key}.png", dpi=200)
        plt.close(figure)

    # Class distribution
    counts = data[TARGET].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(["No failure", "Machine failure"], counts.values)
    axis.set_title("AI4I 2020 Machine-Failure Class Distribution")
    axis.set_ylabel("Number of observations")
    for index, value in enumerate(counts.values):
        axis.text(index, value + 80, f"{value:,}\n({value / len(data) * 100:.2f}%)", ha="center")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "figure_2_class_distribution.png", dpi=200)
    plt.close(figure)

    # ROC curves
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, scores in [
        ("Random Forest", rf_scores),
        ("SVM", svm_scores),
        ("LSTM", lstm_scores),
    ]:
        false_positive, true_positive, _ = roc_curve(y_test, scores)
        axis.plot(
            false_positive,
            true_positive,
            label=f"{name} (AUC={roc_auc_score(y_test, scores):.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "roc_curves.png", dpi=200)
    plt.close(figure)

    # RF feature importance and SHAP summary
    rf_transformer = rf_final.named_steps["pre"]
    rf_classifier = rf_final.named_steps["model"]
    transformed_train = rf_transformer.transform(x_train)
    transformed_test = rf_transformer.transform(x_test)
    feature_names = clean_feature_names(rf_transformer.get_feature_names_out())
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": rf_classifier.feature_importances_}
    ).sort_values("Importance", ascending=False)
    importance.to_csv(TABLE_DIR / "random_forest_feature_importance.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    axis.set_xlabel("Impurity-based importance")
    axis.set_title("Random Forest Feature Importance")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close(figure)

    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = np.asarray(explainer.shap_values(transformed_test[:200]))[:, :, 1]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        transformed_test[:200],
        feature_names=feature_names,
        show=False,
    )
    plt.title("Random Forest SHAP Summary")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary_random_forest.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Training complete. Results saved in outputs/ and models/.")


if __name__ == "__main__":
    main()
