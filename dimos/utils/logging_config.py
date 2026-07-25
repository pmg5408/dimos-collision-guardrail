"""Local stub of dimOS's structured logger so the guardrail runs standalone."""

from __future__ import annotations

import logging
from typing import Any


class _StructuredLogger:
    """Wraps stdlib logging so ``info("msg", key=val)``-style calls work."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @staticmethod
    def _format(message: str, fields: dict[str, Any]) -> str:
        if not fields:
            return message
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        return f"{message} | {rendered}"

    def debug(self, message: str, **fields: Any) -> None:
        self._logger.debug(self._format(message, fields))

    def info(self, message: str, **fields: Any) -> None:
        self._logger.info(self._format(message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self._logger.warning(self._format(message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self._logger.error(self._format(message, fields))

    def exception(self, message: str, **fields: Any) -> None:
        self._logger.exception(self._format(message, fields))


def setup_logger(name: str = "dimos") -> _StructuredLogger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return _StructuredLogger(logger)
