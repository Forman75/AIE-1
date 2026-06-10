from __future__ import annotations

import numpy as np

from src.data.generate import generate_dataset
from src.data.loader import split_data
from src.models.registry import build_model, build_registry
from src.models.trainer import build_pipeline, compute_metrics, evaluate_on_test


def test_build_model_known_types() -> None:
    for mtype in ("logistic_regression", "random_forest", "gradient_boosting"):
        model = build_model(mtype, {})
        assert model is not None


def test_build_model_unknown_type_raises() -> None:
    try:
        build_model("magic_model", {})
        assert False, "Ожидалась ValueError"
    except ValueError:
        pass


def test_build_registry_respects_enabled_flag() -> None:
    config = {
        "a": {"type": "logistic_regression", "enabled": True, "params": {}},
        "b": {"type": "random_forest", "enabled": False, "params": {}},
    }
    registry = build_registry(config)
    assert "a" in registry and "b" not in registry


def test_compute_metrics_range() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.3, 0.7, 0.6, 0.2])
    metrics = compute_metrics(y_true, y_proba)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} вне [0,1]: {value}"
    assert "roc_auc" in metrics and "pr_auc" in metrics


def test_pipeline_trains_and_predicts_probabilities() -> None:
    df = generate_dataset(n_customers=1500, random_state=42)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)

    pipeline = build_pipeline(build_model("logistic_regression", {"max_iter": 500}))
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert len(proba) == len(X_test)


def test_trained_model_beats_random() -> None:
    df = generate_dataset(n_customers=2500, random_state=42)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)

    pipeline = build_pipeline(build_model("gradient_boosting", {"random_state": 42}))
    pipeline.fit(X_train, y_train)
    metrics = evaluate_on_test(pipeline, X_test, y_test)
    assert metrics["roc_auc"] > 0.7, f"Слишком низкий ROC-AUC: {metrics['roc_auc']}"
