from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generate import generate_dataset
from src.utils.config import DATA_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

NUMERIC_FEATURES: list[str] = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]

CATEGORICAL_FEATURES: list[str] = [
    "partner",
    "dependents",
    "contract",
    "internet_service",
    "online_security",
    "tech_support",
    "streaming_tv",
    "payment_method",
    "paperless_billing",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"


def get_dataset(
    n_customers: int = 7000,
    random_state: int = 42,
    target_churn_rate: float = 0.26,
) -> pd.DataFrame:
    logger.info("Генерация датасета: n=%d, seed=%d", n_customers, random_state)
    df = generate_dataset(
        n_customers=n_customers,
        random_state=random_state,
        target_churn_rate=target_churn_rate,
    )
    logger.info(
        "Датасет готов: %d строк, доля оттока %.3f",
        len(df), df[TARGET_COLUMN].mean(),
    )
    return df


def save_sample(df: pd.DataFrame, n: int = 200, path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "sample_customers.csv")
    sample, _ = train_test_split(
        df, train_size=n, stratify=df[TARGET_COLUMN], random_state=42,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    logger.info("Демонстрационная выборка сохранена: %s (%d строк)", path, len(sample))
    return path


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    # стратификация по target, чтобы баланс классов сохранялся
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    logger.info(
        "Разбиение: train=%d, test=%d (отток train=%.3f, test=%.3f)",
        len(X_train), len(X_test), y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test
