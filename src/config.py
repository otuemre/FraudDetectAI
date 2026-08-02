from pathlib import Path

import yaml

from src.logger import get_error_logger, log_exception

error_logger = get_error_logger()

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise log_exception(error_logger, e) from e


CONFIG = load_config()
