from __future__ import annotations

from src.service.app import create_app
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    app = create_app()
    logger.info(
        "Запуск сервиса на http://%s:%d",
        settings.service_host, settings.service_port,
    )
    app.run(
        host=settings.service_host,
        port=settings.service_port,
        debug=False,
    )


if __name__ == "__main__":
    main()
