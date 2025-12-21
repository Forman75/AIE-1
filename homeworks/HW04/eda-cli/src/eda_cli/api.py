from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Импортируем функции из твоего ядра (core.py)
from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис для оценки готовности датасета к обучению модели. "
        "Использует эвристики качества данных из eda-cli."
    ),
    docs_url="/docs",
    redoc_url=None,
)

# ---------- Модели запросов/ответов ----------

class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ..., ge=0.0, le=1.0, description="Максимальная доля пропусков среди всех колонок (0..1)"
    )
    numeric_cols: int = Field(..., ge=0, description="Количество числовых колонок")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных колонок")


class QualityResponse(BaseModel):
    """Ответ модели качества датасета."""
    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели"
    )
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Интегральная оценка качества данных (0..1)"
    )
    message: str = Field(..., description="Человекочитаемое пояснение решения")
    latency_ms: float = Field(..., ge=0.0, description="Время обработки запроса на сервере, миллисекунды")
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)"
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны"
    )


class FlagsResponse(BaseModel):
    """Ответ для кастомного эндпоинта с сырыми флагами."""
    flags: dict[str, Any] = Field(..., description="Полный словарь флагов качества из core.py")


# ---------- Системный эндпоинт ----------

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------

@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества (без чтения самого файла).
    """
    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Эвристики для примера (без данных)
    score -= req.max_missing_share
    if req.n_rows < 1000:
        score -= 0.2
    if req.n_cols > 100:
        score -= 0.1
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "Данных достаточно (по мета-информации)."
    else:
        message = "Качество данных сомнительно (по мета-информации)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: Реальная оценка через ядро ----------

@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    и возвращает финальную оценку (True/False) и скор.
    """
    start = perf_counter()

    # Валидация
    if file.filename and not file.filename.lower().endswith(".csv"):
         pass 

    try:
        # Читаем CSV в DataFrame
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл пуст.")

    # 1. Используем EDA-ядро (core.py)
    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # 2. Считаем флаги. Передаем df, чтобы работали проверки дубликатов и констант
    flags_all = compute_quality_flags(summary, missing_df, df=df)

    # 3. Формируем ответ
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV подходит для модели."
    else:
        message = "CSV требует чистки."

    latency_ms = (perf_counter() - start) * 1000.0

    # Для ответа QualityResponse оставляем только простые булевы флаги
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool) and key != "id_column_exists" # Фильтруем лишнее если надо
    }

    # Размеры
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows, n_cols = df.shape

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv: КАСТОМНЫЙ ЭНДПОИНТ (Вариант A) ----------

@app.post(
    "/quality-flags-from-csv",
    response_model=FlagsResponse,
    tags=["custom"],
    summary="Получение полного набора сырых флагов качества (для отладки)",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> FlagsResponse:
    """
    Кастомный эндпоинт, требуемый в HW04.
    Возвращает ПОЛНЫЙ словарь флагов (включая списки константных колонок,
    дубликатов и т.д.), который генерирует ядро.
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {exc}")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вызываем с df, чтобы получить 'constant_columns', 'suspicious_id_duplicates' и т.д.
    all_flags = compute_quality_flags(summary, missing_df, df=df)

    return FlagsResponse(flags=all_flags)
