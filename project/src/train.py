from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np

from src.data.loader import (
    FEATURE_COLUMNS,
    get_dataset,
    save_sample,
    split_data,
)
from src.models.plots import (
    plot_churn_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.models.registry import build_registry
from src.models.tracking import ExperimentTracker
from src.models.trainer import (
    results_to_dataframe,
    run_experiments,
    select_best_model,
)
from src.utils.config import ARTIFACTS_DIR, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_importances(pipeline) -> tuple[list[str], np.ndarray] | None:
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    names = list(pre.get_feature_names_out())

    if hasattr(clf, "feature_importances_"):
        return names, np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return names, np.abs(np.asarray(clf.coef_).ravel())
    return None


def main() -> None:
    settings = get_settings()
    data_cfg = settings.data
    train_cfg = settings.training

    logger.info("=" * 60)
    logger.info("Обучение моделей")
    logger.info("=" * 60)

    df = get_dataset(
        n_customers=data_cfg["n_customers"],
        random_state=data_cfg["random_state"],
        target_churn_rate=data_cfg["target_churn_rate"],
    )
    save_sample(df, n=data_cfg["sample_size"])

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    registry = build_registry(train_cfg["models"])
    cv_cfg = train_cfg["cross_validation"]
    results = run_experiments(
        registry, X_train, y_train, X_test, y_test,
        cv_splits=cv_cfg["n_splits"],
        cv_random_state=cv_cfg["random_state"],
    )

    selection_metric = train_cfg["selection_metric"]
    best_name = select_best_model(results, metric=selection_metric)
    best_pipeline = results[best_name]["pipeline"]

    tracker = ExperimentTracker()
    tracker.log_params(
        n_customers=data_cfg["n_customers"],
        random_state=data_cfg["random_state"],
        test_size=data_cfg["test_size"],
        cv_splits=cv_cfg["n_splits"],
        selection_metric=selection_metric,
        feature_columns=FEATURE_COLUMNS,
    )
    for name, res in results.items():
        tracker.log_model_result(name, {"cv": res["cv"], "test": res["test"]})
    tracker.set_selected(best_name)
    tracker.save()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "pipeline": best_pipeline,
        "metadata": {
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "selection_metric": selection_metric,
            "cv_metrics": results[best_name]["cv"],
            "test_metrics": results[best_name]["test"],
        },
    }
    joblib.dump(model_artifact, settings.model_path)
    logger.info("Финальная модель сохранена: %s", settings.model_path)

    settings.metrics_path.write_text(
        json.dumps(model_artifact["metadata"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    comparison = results_to_dataframe(results)
    comparison.to_csv(settings.comparison_path, index=False)
    logger.info("Сравнение моделей сохранено: %s", settings.comparison_path)
    logger.info("\n%s", comparison.to_string(index=False))

    plots_dir = ARTIFACTS_DIR / "plots"
    plot_churn_distribution(df, plots_dir / "churn_distribution.png")
    plot_roc_curves(results, X_test, y_test, plots_dir / "roc_curves.png")
    plot_confusion_matrix(
        best_pipeline, X_test, y_test, plots_dir / "confusion_matrix.png"
    )
    importances = _extract_importances(best_pipeline)
    if importances is not None:
        plot_feature_importance(
            importances[0], importances[1],
            plots_dir / "feature_importance.png",
        )

    logger.info("=" * 60)
    logger.info("Обучение завершено. Финальная модель: %s", best_name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
