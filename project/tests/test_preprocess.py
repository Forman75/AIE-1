from __future__ import annotations

import numpy as np

from src.data.generate import generate_dataset
from src.data.loader import FEATURE_COLUMNS
from src.features.preprocess import build_preprocessor, get_feature_names


def _sample_X(n: int = 400):
    df = generate_dataset(n_customers=n, random_state=5)
    return df[FEATURE_COLUMNS]


def test_preprocessor_fit_transform_runs() -> None:
    X = _sample_X()
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)


def test_preprocessor_removes_missing_values() -> None:
    X = _sample_X()
    assert X["total_charges"].isna().sum() > 0
    transformed = build_preprocessor().fit_transform(X)
    assert not np.isnan(transformed).any()


def test_feature_names_match_columns() -> None:
    X = _sample_X()
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    names = get_feature_names(pre)
    assert len(names) == transformed.shape[1]


def test_handles_unknown_category() -> None:
    X = _sample_X()
    pre = build_preprocessor().fit(X)
    X_new = X.head(3).copy()
    X_new.loc[X_new.index[0], "contract"] = "Unknown contract type"
    transformed = pre.transform(X_new)
    assert transformed.shape[0] == 3
