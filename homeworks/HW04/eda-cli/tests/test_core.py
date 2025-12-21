import pandas as pd
from eda_cli.core import (
    compute_quality_flags,
    has_constant_columns,
    has_high_cardinality_categoricals,
    has_suspicious_id_duplicates,
    has_many_zero_values,
    DatasetSummary,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
            "user_id": [1, 2, 3, 2],
            "constant_col": [1, 1, 1, 1],
            "high_card_col": ['a', 'b', 'c', 'd'],
            "zero_col": [0, 0, 0, 0],  # Столбец с нулями
        }
    )


def test_has_constant_columns():
    df = _sample_df()
    result = has_constant_columns(df)
    assert set(result) == {"constant_col", "zero_col"}  # Ожидаем, что constant_col будет постоянным


def test_has_high_cardinality_categoricals():
    df = _sample_df()
    result = has_high_cardinality_categoricals(df, threshold=2)
    assert result == ["high_card_col"]  # Ожидаем, что high_card_col будет иметь высокую кардинальность


def test_has_suspicious_id_duplicates():
    df = _sample_df()
    result = has_suspicious_id_duplicates(df, id_column="user_id")
    assert result == True  # Ожидаем, что есть дубликаты в user_id


def test_has_many_zero_values():
    df = _sample_df()
    result = has_many_zero_values(df, threshold=0.5)
    assert result["zero_col"] is True  # Ожидаем, что zero_col имеет больше 50% нулевых значений


def test_compute_quality_flags():
    df = _sample_df()
    summary = DatasetSummary(
        n_rows=4,
        n_cols=6,
        columns=[
            {
                "name": "age",
                "dtype": "float64",
                "non_null": 3,
                "missing": 1,
                "missing_share": 0.25,
                "unique": 3,
                "example_values": ["1", "2", "3"],
                "is_numeric": True,
            },
            {
                "name": "city",
                "dtype": "object",
                "non_null": 3,
                "missing": 1,
                "missing_share": 0.25,
                "unique": 2,
                "example_values": ["A", "B"],
                "is_numeric": False,
            },
            # другие колонки
        ]
    )

    missing_df = pd.DataFrame({
        "missing_count": [1, 1, 0, 0],
        "missing_share": [0.25, 0.25, 0.0, 0.0],
    }, index=["age", "city", "height", "user_id"])

    flags = compute_quality_flags(summary, missing_df, df=df)

    assert "constant_columns" in flags
    assert "high_cardinality_categoricals" in flags
    assert "suspicious_id_duplicates" in flags
    assert "many_zero_values" in flags
    assert "quality_score" in flags
    assert 0.0 <= flags["quality_score"] <= 1.0  # Ожидаем, что quality_score в пределах [0, 1]
