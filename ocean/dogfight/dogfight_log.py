"""File-based logging for Dogfight training diagnostics."""

import logging
import os
from datetime import datetime


logger = logging.getLogger("dogfight")
logger.setLevel(logging.DEBUG)


def init_log(log_dir="league/logs", run_name=None):
    """Create a timestamped Dogfight log file and attach it to the logger."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{run_name}_{timestamp}.log" if run_name else f"dogfight_{timestamp}.log"
    path = os.path.join(log_dir, filename)

    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.info("[ROUND] log_started path=%s", path)
    return path


log = logger.info
