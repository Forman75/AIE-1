from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


def load_dotenv(env_path: Path | None = None) -> None:
    path = env_path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Не найден конфигурационный файл: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:

    project: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)

    training: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)

    log_level: str = "INFO"
    service_host: str = "0.0.0.0"
    service_port: int = 8000
    api_token: str | None = None

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        base = load_yaml("config.yaml")
        training = load_yaml("training.yaml")
        inference = load_yaml("inference.yaml")

        return cls(
            project=base.get("project", {}),
            data=base.get("data", {}),
            paths=base.get("paths", {}),
            training=training,
            inference=inference,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            service_host=os.getenv("SERVICE_HOST", "0.0.0.0"),
            service_port=int(os.getenv("SERVICE_PORT", "8000")),
            api_token=os.getenv("API_TOKEN") or None,
        )


    @property
    def model_path(self) -> Path:
        return ARTIFACTS_DIR / self.paths.get("model_file", "model.joblib")

    @property
    def metrics_path(self) -> Path:
        return ARTIFACTS_DIR / self.paths.get("metrics_file", "metrics.json")

    @property
    def comparison_path(self) -> Path:
        return ARTIFACTS_DIR / self.paths.get("comparison_file", "model_comparison.csv")


def get_settings() -> Settings:
    return Settings.load()
