from __future__ import annotations

from typing import Any

from src.data.loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ALLOWED_VALUES: dict[str, set[str]] = {
    "partner": {"Yes", "No"},
    "dependents": {"Yes", "No"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "online_security": {"Yes", "No", "No internet service"},
    "tech_support": {"Yes", "No", "No internet service"},
    "streaming_tv": {"Yes", "No", "No internet service"},
    "payment_method": {
        "Electronic check", "Mailed check", "Bank transfer", "Credit card",
    },
    "paperless_billing": {"Yes", "No"},
}

NUMERIC_SPEC: dict[str, dict[str, Any]] = {
    "senior_citizen": {"type": int, "min": 0, "max": 1, "nullable": False},
    "tenure": {"type": int, "min": 0, "max": 100, "nullable": False},
    "monthly_charges": {"type": float, "min": 0.0, "max": 1000.0, "nullable": False},
    "total_charges": {"type": float, "min": 0.0, "max": 100000.0, "nullable": True},
    "num_support_calls": {"type": int, "min": 0, "max": 100, "nullable": False},
}

EXAMPLE_REQUEST: dict[str, Any] = {
    "senior_citizen": 0,
    "tenure": 2,
    "monthly_charges": 89.5,
    "total_charges": 178.6,
    "num_support_calls": 4,
    "partner": "No",
    "dependents": "No",
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "payment_method": "Electronic check",
    "paperless_billing": "Yes",
}


class ValidationError(Exception):

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_customer(payload: Any) -> dict[str, Any]:
    errors: list[str] = []

    if not isinstance(payload, dict):
        raise ValidationError(["Тело запроса должно быть JSON-объектом"])

    cleaned: dict[str, Any] = {}

    for field, spec in NUMERIC_SPEC.items():
        if field not in payload or payload[field] is None:
            if spec["nullable"]:
                cleaned[field] = None
                continue
            errors.append(f"Поле '{field}' обязательно")
            continue
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"Поле '{field}' должно быть числом")
            continue
        value = spec["type"](value)
        if not (spec["min"] <= value <= spec["max"]):
            errors.append(
                f"Поле '{field}' вне диапазона [{spec['min']}, {spec['max']}]"
            )
            continue
        cleaned[field] = value

    for field in CATEGORICAL_FEATURES:
        if field not in payload or payload[field] is None:
            errors.append(f"Поле '{field}' обязательно")
            continue
        value = payload[field]
        if value not in ALLOWED_VALUES[field]:
            allowed = sorted(ALLOWED_VALUES[field])
            errors.append(
                f"Поле '{field}'={value!r} недопустимо. Допустимо: {allowed}"
            )
            continue
        cleaned[field] = value

    expected = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    unknown = set(payload) - expected
    if unknown:
        pass

    if errors:
        raise ValidationError(errors)
    return cleaned
