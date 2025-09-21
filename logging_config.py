import logging
import sys
import re
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import config


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from logs"""
    
    def __init__(self):
        super().__init__()
        # Patterns for sensitive data
        self.patterns = [
            (re.compile(r'Bearer [A-Za-z0-9\-_]+'), 'Bearer [REDACTED]'),
            (re.compile(r'"api_key":\s*"[^"]+"'), '"api_key": "[REDACTED]"'),
            (re.compile(r'"password":\s*"[^"]+"'), '"password": "[REDACTED]"'),
            (re.compile(r'"token":\s*"[^"]+"'), '"token": "[REDACTED]"'),
            (re.compile(r'sk-[A-Za-z0-9]{48}'), 'sk-[REDACTED]'),
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL_REDACTED]'),
            (re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), '[CARD_REDACTED]'),
        ]
    
    def filter(self, record):
        if hasattr(record, 'msg'):
            message = str(record.msg)
            for pattern, replacement in self.patterns:
                message = pattern.sub(replacement, message)
            record.msg = message
        return True

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create sensitive data filter
    sensitive_filter = SensitiveDataFilter()

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
        file_handler.addFilter(sensitive_filter)
        handlers.append(file_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Changed from DEBUG to INFO
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        handlers.append(console_handler)

    logging.basicConfig(
        level=logging.INFO,  # Always use INFO level
        handlers=handlers, 
        force=True)
    
    # Silence noisy loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("databases").setLevel(logging.WARNING)

setup_logging()
