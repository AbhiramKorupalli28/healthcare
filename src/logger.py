"""Processing log: every pipeline step is written to output/logs/processing.log."""
import logging
from config import LOG_DIR

_LOG_FILE = LOG_DIR / "processing.log"


def get_logger(name: str = "healthcare") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
    return logger


def tail_log(n: int = 30) -> list[str]:
    """Last n lines of the processing log (empty list if nothing has run yet)."""
    if not _LOG_FILE.exists():
        return []
    lines = _LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]
