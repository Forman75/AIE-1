from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import ARTIFACTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:

    def __init__(self, experiments_dir: Path | None = None) -> None:
        self.dir = experiments_dir or (ARTIFACTS_DIR / "experiments")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        self._record: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "models": {},
            "selected_model": None,
        }
        logger.info("Эксперимент начат: %s", self.run_id)

    def log_params(self, **params: Any) -> None:
        self._record["params"].update(params)

    def log_model_result(self, name: str, result: dict[str, Any]) -> None:
        self._record["models"][name] = result
        logger.info("Записаны результаты модели '%s'", name)

    def set_selected(self, name: str) -> None:
        self._record["selected_model"] = name

    def save(self) -> Path:
        self._record["finished_at"] = datetime.now(timezone.utc).isoformat()
        path = self.dir / f"{self.run_id}.json"
        path.write_text(
            json.dumps(self._record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Эксперимент сохранён: %s", path)
        return path

    @property
    def record(self) -> dict[str, Any]:
        return self._record
