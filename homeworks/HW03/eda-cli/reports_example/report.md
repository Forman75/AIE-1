# EDA-отчёт

Исходный файл: `example.csv`

Строк: **36**, столбцов: **14**

## Параметры генерации отчёта

- max_hist_columns = **6**
- top_k_categories = **5**
- min_missing_share = **10%**

## Качество данных (эвристики)

- quality_score: **0.54**
- max_missing_share: **5.56%**

### Дополнительные эвристики (HW03)

- constant_columns: нет
- high_cardinality_categoricals (threshold=50): нет
- suspicious_id_duplicates (id_column=user_id): True
- many_zero_values (threshold=50%): churned

## Пропуски

См. `missing.csv` и `missing_matrix.png`.

### Проблемные колонки по пропускам (HW03)

Колонок с missing_share >= 10% не найдено.

## Корреляция

См. `correlation.csv` и `correlation_heatmap.png`.

## Категориальные признаки

Сохранены top-5 значения (см. `top_categories/`).

## Гистограммы

См. `hist_*.png`.
