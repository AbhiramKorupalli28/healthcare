"""Shared Streamlit state: the active dataset, the cached Spark session and cached analyses.

Caches are keyed by the cleaned-data path, which includes the file fingerprint, so a
new upload produces new results and an unchanged file reuses the old ones.
"""
import json
from pathlib import Path

import streamlit as st

from config import DEFAULT_DATASET
from src.data_loader import file_fingerprint
from src.logger import get_logger

log = get_logger("app")


def active_dataset_path() -> Path:
    return Path(st.session_state.get("active_dataset", str(DEFAULT_DATASET)))


def set_active_dataset(path) -> None:
    st.session_state["active_dataset"] = str(path)


@st.cache_resource(show_spinner="Starting the Spark session...")
def spark_session():
    from src.spark_engine import get_spark
    return get_spark()


def active_fingerprint() -> str:
    return file_fingerprint(active_dataset_path())


def ensure_cleaned(storage: str = "LOCAL", force: bool = False) -> dict:
    """Cleaned-data statistics for the active dataset.

    A file already cleaned for this fingerprint and storage mode is reused without
    starting Spark. Otherwise the Spark pipeline runs and shows a progress bar.
    """
    from src.pipeline import load_stats, run_cleaning_pipeline
    path = active_dataset_path()
    fp = active_fingerprint()
    if not force:
        found = load_stats(path.name, fp, storage)
        if found:
            return found | {"status": "reused"}
    bar = st.progress(0.0, text="Preparing the dataset with Spark")
    stats = run_cleaning_pipeline(spark_session(), str(path), fingerprint=fp,
                                  source_name=path.name, storage=storage, force=force,
                                  progress=lambda p, m: bar.progress(min(max(p, 0.0), 1.0), text=m))
    bar.empty()
    return stats


@st.cache_data(show_spinner="Running Spark aggregations...")
def cached_frame(key: str, path: str, builder: str, args_json: str = "{}"):
    """Run a named analysis from src.analytics over `path` with Spark, and cache the result.

    `key` only separates cache entries; `builder` is the function name in src.analytics.
    """
    from src import analytics
    fn = getattr(analytics, builder)
    return fn(spark_session(), path, **json.loads(args_json))
