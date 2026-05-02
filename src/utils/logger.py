import json
import logging
from datetime import datetime
from pathlib import Path

from src.utils.paths import get_project_layout


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_CURRENT_RUN_ID = get_run_id()


def setup_logger(name: str, level: str = "INFO", run_id: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if run_id is None:
        run_id = _CURRENT_RUN_ID

    log_dir = get_project_layout().logs_runs / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / f"{name}.jsonl", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)

    return logger


def get_current_run_dir() -> Path:
    return get_project_layout().logs_runs / _CURRENT_RUN_ID
