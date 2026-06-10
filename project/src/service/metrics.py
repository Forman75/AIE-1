from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


class MetricsCollector:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._total_requests = 0
        self._requests_by_path: dict[str, int] = defaultdict(int)
        self._status_codes: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._predictions = {"churn": 0, "no_churn": 0}
        self._latency_sum_ms = 0.0
        self._latency_count = 0

    def record_request(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._total_requests += 1
            self._requests_by_path[path] += 1
            self._status_codes[str(status_code)] += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            self._errors[kind] += 1

    def record_prediction(self, prediction: int) -> None:
        with self._lock:
            key = "churn" if prediction == 1 else "no_churn"
            self._predictions[key] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = (
                self._latency_sum_ms / self._latency_count
                if self._latency_count else 0.0
            )
            uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._total_requests,
                "requests_by_path": dict(self._requests_by_path),
                "status_codes": dict(self._status_codes),
                "errors": dict(self._errors),
                "predictions": dict(self._predictions),
                "avg_latency_ms": round(avg_latency, 2),
            }
