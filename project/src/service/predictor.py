from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.loader import FEATURE_COLUMNS
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_FEATURE_LABELS: dict[str, str] = {
    "tenure": "Срок обслуживания (мес.)",
    "contract": "Тип контракта",
    "monthly_charges": "Ежемесячные траты",
    "total_charges": "Суммарные траты",
    "internet_service": "Тип интернет-услуги",
    "tech_support": "Техподдержка подключена",
    "online_security": "Онлайн-безопасность подключена",
    "payment_method": "Способ оплаты",
    "paperless_billing": "Безбумажный биллинг",
    "senior_citizen": "Пожилой клиент",
    "partner": "Есть супруг(а)",
    "dependents": "Есть иждивенцы",
    "streaming_tv": "Стриминг ТВ",
    "num_support_calls": "Число обращений в поддержку",
}


class ModelNotLoadedError(RuntimeError):
    pass


class ChurnPredictor:

    def __init__(self, model_path: Path | None = None) -> None:
        settings = get_settings()
        self._model_path = model_path or settings.model_path
        self._inference_cfg = settings.inference
        self._pipeline = None
        self._metadata: dict[str, Any] = {}
        self._feature_importance: dict[str, float] = {}


    def load(self) -> "ChurnPredictor":
        if not self._model_path.exists():
            raise ModelNotLoadedError(
                f"Файл модели не найден: {self._model_path}. "
                f"Сначала обучите модель: python -m src.train"
            )
        artifact = joblib.load(self._model_path)
        self._pipeline = artifact["pipeline"]
        self._metadata = artifact["metadata"]
        self._feature_importance = self._compute_grouped_importance()
        logger.info(
            "Модель загружена: %s (обучена %s)",
            self._metadata.get("model_name"),
            self._metadata.get("trained_at"),
        )
        return self

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.is_ready:
            raise ModelNotLoadedError("Модель не загружена. Вызовите load().")

        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        probability = float(self._pipeline.predict_proba(row)[0, 1])

        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": int(probability >= 0.5),
            "risk_category": self._risk_category(probability),
            "key_factors": self._key_factors(features),
            "model_name": self._metadata.get("model_name"),
        }


    def _risk_category(self, probability: float) -> str:
        thresholds = self._inference_cfg.get("risk_thresholds", {})
        low_max = thresholds.get("low_max", 0.35)
        high_min = thresholds.get("high_min", 0.65)
        if probability < low_max:
            return "low"
        if probability < high_min:
            return "medium"
        return "high"

    def _compute_grouped_importance(self) -> dict[str, float]:
        clf = self._pipeline.named_steps["classifier"]
        pre = self._pipeline.named_steps["preprocessor"]
        names = list(pre.get_feature_names_out())

        if hasattr(clf, "feature_importances_"):
            raw = np.asarray(clf.feature_importances_)
        elif hasattr(clf, "coef_"):
            raw = np.abs(np.asarray(clf.coef_).ravel())
        else:
            return {}

        grouped: dict[str, float] = defaultdict(float)
        for name, value in zip(names, raw):
            stripped = name.split("__", 1)[-1]
            original = next(
                (f for f in FEATURE_COLUMNS if stripped.startswith(f)),
                stripped,
            )
            grouped[original] += float(value)
        return dict(grouped)

    def _key_factors(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        top_k = int(self._inference_cfg.get("top_factors", 5))
        ranked = sorted(
            self._feature_importance.items(),
            key=lambda kv: kv[1], reverse=True,
        )[:top_k]
        total = sum(self._feature_importance.values()) or 1.0
        return [
            {
                "feature": feat,
                "label": _FEATURE_LABELS.get(feat, feat),
                "value": features.get(feat),
                "importance": round(importance / total, 4),
            }
            for feat, importance in ranked
        ]
