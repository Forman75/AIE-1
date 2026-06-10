from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(model_type: str, params: dict[str, Any]) -> ClassifierMixin:
    if model_type not in _BUILDERS:
        raise ValueError(
            f"Неизвестный тип модели: {model_type!r}. "
            f"Доступные: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[model_type](**params)


def build_registry(models_config: dict[str, Any]) -> dict[str, ClassifierMixin]:
    registry: dict[str, ClassifierMixin] = {}
    for name, spec in models_config.items():
        if not spec.get("enabled", True):
            continue
        registry[name] = build_model(spec["type"], spec.get("params", {}))
    return registry
