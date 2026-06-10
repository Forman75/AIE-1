from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor
from src.utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_pred = (y_proba >= 0.5).astype(int)
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }
    for name, fn in _LABEL_METRICS.items():
        kwargs = {"zero_division": 0} if name != "accuracy" else {}
        metrics[name] = float(fn(y_true, y_pred, **kwargs))
    return {k: round(v, 4) for k, v in metrics.items()}


def build_pipeline(model: ClassifierMixin) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    return compute_metrics(y.to_numpy(), oof_proba)


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test.to_numpy(), proba)


def run_experiments(
    registry: dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
    cv_random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, model in registry.items():
        logger.info("=== Модель: %s ===", name)
        pipeline = build_pipeline(model)

        cv_metrics = cross_validate_model(
            pipeline, X_train, y_train, cv_splits, cv_random_state,
        )
        logger.info("  CV:   %s", _fmt(cv_metrics))

        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_on_test(pipeline, X_test, y_test)
        logger.info("  Test: %s", _fmt(test_metrics))

        results[name] = {
            "cv": cv_metrics,
            "test": test_metrics,
            "pipeline": pipeline,
        }
    return results


def select_best_model(
    results: dict[str, dict[str, Any]],
    metric: str = "roc_auc",
) -> str:
    # выбираем по CV, тест оставляем для финальной проверки
    best = max(results.items(), key=lambda kv: kv[1]["cv"][metric])
    logger.info(
        "Выбрана финальная модель: '%s' (CV %s=%.4f)",
        best[0], metric, best[1]["cv"][metric],
    )
    return best[0]


def results_to_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        row: dict[str, Any] = {"model": name}
        for split in ("cv", "test"):
            for metric, value in res[split].items():
                row[f"{split}_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)


def _fmt(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
