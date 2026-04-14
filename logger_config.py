"""
Logging setup — writes logs to BOTH:
  1. Console (stdout) — to see live in terminal
  2. File (logs/app.log) — persistent log file saved on disk
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

MAX_BYTES = 2 * 1024 * 1024   # 2MB per file
BACKUP_COUNT = 3               # keeps app.log, app.log.1, app.log.2


def setup_logging(level=logging.DEBUG):
    """
    Configures root logger with:
    - Console handler  → prints to terminal
    - File handler     → writes to logs/app.log (auto-rotates at 2MB)
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging initialised | level=%s | file=%s",
        logging.getLevelName(level), LOG_FILE
    )