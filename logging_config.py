import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import config


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers = []

    if config.ENV_STATE == "prod":
        file_handler = RotatingFileHandler(
            log_dir / "semantica.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    logging.basicConfig(
        level=logging.DEBUG if config.ENV_STATE != "prod" else logging.INFO, 
        handlers=handlers, 
        force=True)
    
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

setup_logging()
