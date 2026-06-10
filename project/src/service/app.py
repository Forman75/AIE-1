from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from flask import Flask, g, jsonify, request

from src.service.metrics import MetricsCollector
from src.service.predictor import ChurnPredictor, ModelNotLoadedError
from src.service.schemas import EXAMPLE_REQUEST, ValidationError, validate_customer
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    settings = get_settings()
    metrics = MetricsCollector()

    predictor = ChurnPredictor()
    try:
        predictor.load()
    except ModelNotLoadedError as exc:
        logger.warning("Модель не загружена: %s", exc)


    @app.before_request
    def _start_timer() -> None:
        g.start_time = time.perf_counter()

    @app.after_request
    def _log_request(response):
        latency_ms = (time.perf_counter() - getattr(g, "start_time", time.perf_counter())) * 1000
        metrics.record_request(request.path, response.status_code, latency_ms)
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method, request.path, response.status_code, latency_ms,
        )
        return response


    @app.get("/")
    def index() -> Any:
        return jsonify(
            {
                "service": settings.project.get("name", "churn-prediction"),
                "description": settings.project.get("description", ""),
                "version": "1.0.0",
                "endpoints": {
                    "GET /health": "проверка работоспособности",
                    "GET /metrics": "счётчики наблюдаемости",
                    "POST /predict": "прогноз оттока по профилю клиента",
                },
                "predict_example_request": EXAMPLE_REQUEST,
            }
        )

    @app.get("/health")
    def health() -> Any:
        ready = predictor.is_ready
        body = {
            "status": "ok" if ready else "degraded",
            "model_loaded": ready,
            "model_name": predictor.metadata.get("model_name"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return jsonify(body), (200 if ready else 503)

    @app.get("/metrics")
    def get_metrics() -> Any:
        return jsonify(metrics.snapshot())

    @app.post("/predict")
    def predict() -> Any:
        # опциональная проверка токена
        if settings.api_token:
            token = request.headers.get("X-API-Token")
            if token != settings.api_token:
                metrics.record_error("auth")
                return jsonify({"error": "Неверный или отсутствующий X-API-Token"}), 401

        if not predictor.is_ready:
            metrics.record_error("model_not_loaded")
            return jsonify(
                {"error": "Модель не загружена. Выполните: python -m src.train"}
            ), 503

        try:
            payload = request.get_json(force=True, silent=False)
        except Exception:
            metrics.record_error("bad_json")
            return jsonify({"error": "Тело запроса не является корректным JSON"}), 400

        try:
            features = validate_customer(payload)
        except ValidationError as exc:
            metrics.record_error("validation")
            return jsonify({"error": "Ошибка валидации", "details": exc.errors}), 422

        try:
            result = predictor.predict(features)
        except Exception as exc:
            metrics.record_error("inference")
            logger.exception("Ошибка инференса")
            return jsonify({"error": f"Ошибка инференса: {exc}"}), 500

        metrics.record_prediction(result["churn_prediction"])
        return jsonify(result)

    @app.errorhandler(404)
    def not_found(_: Any) -> Any:
        return jsonify({"error": "Эндпоинт не найден"}), 404

    logger.info("Flask-приложение создано")
    return app
