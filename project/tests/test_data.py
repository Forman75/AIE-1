from __future__ import annotations

import pandas as pd

from src.data.generate import generate_dataset
from src.data.loader import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    split_data,
)


def test_generated_dataset_has_expected_shape() -> None:
    df = generate_dataset(n_customers=500, random_state=1)
    assert len(df) == 500
    expected_cols = set(FEATURE_COLUMNS) | {TARGET_COLUMN, "customer_id"}
    assert expected_cols.issubset(set(df.columns))


def test_churn_rate_in_reasonable_range() -> None:
    df = generate_dataset(n_customers=3000, random_state=42, target_churn_rate=0.26)
    rate = df[TARGET_COLUMN].mean()
    assert 0.18 < rate < 0.34, f"Неожиданная доля оттока: {rate}"


def test_target_is_binary() -> None:
    df = generate_dataset(n_customers=500, random_state=7)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_total_charges_has_missing_values() -> None:
    df = generate_dataset(n_customers=2000, random_state=42)
    assert df["total_charges"].isna().sum() > 0


def test_generation_is_deterministic() -> None:
    df1 = generate_dataset(n_customers=300, random_state=123)
    df2 = generate_dataset(n_customers=300, random_state=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_split_preserves_class_balance() -> None:
    df = generate_dataset(n_customers=2000, random_state=42)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_feature_schema_consistency() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))
    assert len(FEATURE_COLUMNS) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
