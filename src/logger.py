import json
import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
TRAINING_LOG_DIR = LOGS_DIR / "training"
ERROR_LOG_DIR = LOGS_DIR / "errors"

TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON, so each line in the log file is a valid JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now().isoformat(),  # noqa: DTZ005
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # If the message itself is a dict (e.g. from CustomException.to_dict()),
        # merge its fields in directly rather than nesting it as a string
        if isinstance(record.msg, dict):
            log_entry.update(record.msg)
            log_entry["message"] = record.msg.get("message", "")

        return json.dumps(log_entry)


def get_training_logger() -> logging.Logger:
    """One log file per training run, timestamped at creation time."""
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
    log_path = TRAINING_LOG_DIR / f"training_{run_timestamp}.json"

    logger = logging.getLogger(f"training_{run_timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if called twice in same session

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger


def get_error_logger() -> logging.Logger:
    """One log file per day, all exceptions (training or prediction) land here."""
    day_str = datetime.now().strftime("%Y-%m-%d")  # noqa: DTZ005
    log_path = ERROR_LOG_DIR / f"errors_{day_str}.json"

    logger = logging.getLogger("errors")
    logger.setLevel(logging.ERROR)

    # Only add the handler once per day — avoid duplicate handlers on repeated calls
    if not logger.handlers or logger.handlers[0].baseFilename != str(
        log_path.resolve()
    ):
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    return logger
