"""End-to-end cleaning run: read -> standardise -> profile -> clean -> write Parquet -> stats.

Results are cached per dataset fingerprint: running the same file again reuses the
Parquet and run_stats.json already written, so nothing is processed twice.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession

from config import CLEAN_DIR, DATASET_INFO, DEFAULT_DATASET, HDFS_FS_URI, HDFS_PROCESSED_DIR, SPARK_MASTER
from src.codebook import apply_codebook
from src.data_loader import detect_roles, file_info, slug_for
from src.logger import get_logger
from src.preprocessing import clean, consistency_warnings, invalid_breakdown, profile, standardize_columns
from src.spark_engine import read_csv, spark_info

log = get_logger("pipeline")


def _out_dirs(source_name: str, fingerprint: str, storage: str):
    base = f"{slug_for(source_name)}_{fingerprint}" + ("_hdfs" if storage == "HDFS" else "_local")
    local_dir = CLEAN_DIR / base
    if storage == "HDFS":
        remote = f"{HDFS_FS_URI}{HDFS_PROCESSED_DIR}/{base}"
        return local_dir, remote
    return local_dir, str(local_dir)


def run_cleaning_pipeline(spark: SparkSession, source, *, fingerprint: str, source_name: str,
                          storage: str = "LOCAL", force: bool = False, progress=None) -> dict:
    """Clean one dataset with Spark. Returns the run statistics (also saved as run_stats.json)."""
    progress = progress or (lambda pct, msg: None)
    local_dir, out_root = _out_dirs(source_name, fingerprint, storage)
    stats_file = local_dir / "run_stats.json"
    if stats_file.exists() and not force:
        progress(1.0, "Using the cleaned data already produced for this file")
        log.info("reuse cleaned output for %s (%s)", source_name, fingerprint)
        return json.loads(stats_file.read_text(encoding="utf-8")) | {"status": "reused"}

    local_dir.mkdir(parents=True, exist_ok=True)
    timings, t0 = {}, time.time()
    log.info("start cleaning %s from %s storage (%s), Spark master %s",
             source_name, storage, source, SPARK_MASTER)

    progress(0.05, "Reading CSV with Spark")
    t = time.time()
    raw = read_csv(spark, str(source))
    raw_cols = len(raw.columns)
    raw, coded = apply_codebook(raw)           # CDC codes -> readable names and labels (only if present)
    std, col_map = standardize_columns(raw)
    timings["read_csv_s"] = round(time.time() - t, 2)
    log.info("read %d columns; standardised names", raw_cols)

    progress(0.2, "Profiling the raw data (rows, missing values, duplicates)")
    t = time.time()
    before = profile(std)
    timings["profile_before_s"] = round(time.time() - t, 2)
    log.info("before: %d rows, %d missing cells, %d duplicate rows",
             before["rows"], before["missing_cells"], before["duplicate_rows"])

    roles = detect_roles(std.columns)
    log.info("detected roles: %s", ", ".join(sorted(roles)) or "none")

    progress(0.4, "Cleaning: duplicates, types, validity rules")
    t = time.time()
    cleaned, rejected = clean(std, roles)
    cons = consistency_warnings(std, roles)
    cleaned.cache()
    n_clean = cleaned.count()
    n_rejected = rejected.count()
    invalid = invalid_breakdown(rejected)
    timings["clean_s"] = round(time.time() - t, 2)
    log.info("cleaned rows %d, rejected %d", n_clean, n_rejected)

    progress(0.65, "Writing cleaned Parquet")
    t = time.time()
    clean_path = f"{out_root}/cleaned.parquet"
    rej_path = f"{out_root}/rejected.parquet"
    cleaned.write.mode("overwrite").parquet(clean_path)
    rejected.write.mode("overwrite").parquet(rej_path)
    timings["write_parquet_s"] = round(time.time() - t, 2)
    log.info("wrote Parquet to %s", clean_path)

    progress(0.85, "Profiling the cleaned data")
    t = time.time()
    after = profile(cleaned)
    timings["profile_after_s"] = round(time.time() - t, 2)

    info = file_info(source) if not str(source).startswith("hdfs://") else {"size_bytes": None, "size_mb": None}
    dup_removed = before["duplicate_rows"]
    stats = {
        "status": "fresh",
        "dataset": source_name,
        "dataset_title": DATASET_INFO["name"] if source_name == DEFAULT_DATASET.name else source_name,
        "codebook_changes": coded,
        "fingerprint": fingerprint,
        "source": str(source),
        "source_storage": "HDFS" if str(source).startswith("hdfs://") else "LOCAL",
        "storage_mode": storage,
        "file_size_bytes": info.get("size_bytes"),
        "file_size_mb": info.get("size_mb"),
        "raw_columns": raw_cols,
        "column_names_changed": sum(1 for k, v in col_map.items() if k != v),
        "roles_detected": roles,
        "before": before,
        "duplicates_removed": dup_removed,
        "rejected_rows": n_rejected,
        "invalid_breakdown": invalid,
        "consistency_warnings_kept": cons,
        "after": after,
        "reconciles": before["rows"] - dup_removed - n_rejected == n_clean,
        "cleaned_rows": n_clean,
        "cleaned_path": clean_path,
        "rejected_path": rej_path,
        "spark": spark_info(spark),
        "timings_s": timings,
        "total_s": round(time.time() - t0, 2),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    cleaned.unpersist()
    stats_file.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    progress(1.0, "Cleaning finished")
    log.info("finished in %.1f s; reconciles=%s", stats["total_s"], stats["reconciles"])
    return stats


def load_stats(source_name: str, fingerprint: str, storage: str = "LOCAL") -> dict | None:
    base = f"{slug_for(source_name)}_{fingerprint}" + ("_hdfs" if storage == "HDFS" else "_local")
    f = CLEAN_DIR / base / "run_stats.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
