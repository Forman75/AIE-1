from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


def plot_roc_curves(
    results: dict[str, dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        proba = res["pipeline"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = res["test"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые моделей (тестовая выборка)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_confusion_matrix(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path,
) -> None:
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        cm, display_labels=["Остался", "Ушёл"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Матрица ошибок — финальная модель")
    _save(fig, out_path)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    top_k: int = 15,
) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), values, color="#2563eb")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Значимость")
    ax.set_title(f"Топ-{top_k} значимых признаков")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_path)


def plot_churn_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["churn"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["Остался (0)", "Ушёл (1)"], counts.values,
           color=["#22c55e", "#ef4444"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Распределение оттока клиентов")
    ax.set_ylabel("Число клиентов")
    _save(fig, out_path)


def _save(fig: "plt.Figure", out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("График сохранён: %s", out_path)
