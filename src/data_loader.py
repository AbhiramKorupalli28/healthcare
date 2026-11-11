"""Dataset discovery, fingerprints, dynamic column detection and light previews.

Nothing here assumes a schema: columns are matched to logical roles by the
alias table in config.py, and anything not matched is simply not used.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

from config import COLUMN_ALIASES, DATASET_DIR, UPLOAD_DIR


def to_snake(name: str) -> str:
    """'Blood_Pressure Systolic' -> 'blood_pressure_systolic'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return s or "column"


def detect_roles(columns, aliases=None) -> dict:
    """Map logical roles (age, bmi, glucose, ...) to standardised column names found in `columns`.

    Columns are compared after standardisation, so 'Blood_Glucose' and
    'blood glucose' both match the role 'glucose'. Roles with no match are absent.
    """
    aliases = aliases or COLUMN_ALIASES
    present = {to_snake(c) for c in columns}
    roles = {}
    for role, candidates in aliases.items():
        for cand in candidates:
            if cand in present:
                roles[role] = cand
                break
    return roles


def file_fingerprint(path) -> str:
    """Short content fingerprint: size plus SHA-256 of the first and last 8 MB.

    Cheap even for multi-GB files, and changes whenever the file changes.
    """
    path = Path(path)
    size = path.stat().st_size
    chunk = 8 * 1024 * 1024
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def file_info(path) -> dict:
    path = Path(path)
    size = path.stat().st_size
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": size,
        "size_mb": round(size / 1_000_000, 2),
        "fingerprint": file_fingerprint(path),
    }


def preview_pandas(path, nrows: int = 100) -> pd.DataFrame:
    """First rows of a CSV, read by pandas for display only (never for analytics)."""
    return pd.read_csv(path, nrows=nrows)


def slug_for(path) -> str:
    """Filesystem-safe base name for a dataset's outputs."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", Path(path).stem).strip("_").lower() or "dataset"


def list_local_datasets() -> list[Path]:
    """CSV files in dataset/ and dataset/uploads/ (the dashboard's dataset picker)."""
    files = []
    for folder in (DATASET_DIR, UPLOAD_DIR):
        if folder.exists():
            files += sorted(folder.glob("*.csv"))
    return files


def is_hdfs_uri(path) -> bool:
    return str(path).startswith("hdfs://")
