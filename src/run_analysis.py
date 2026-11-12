"""Command-line run of the whole analysis (no dashboard needed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from config import CLEAN_DIR, DEFAULT_DATASET, HDFS_INPUT_DIR, OUTPUT_DIR, SUMMARY_DIR
from src.analytics import disease_view, ml_frame, overview, risk_counts, sql_results
from src.data_loader import detect_roles, file_fingerprint
from src.hdfs_manager import HDFSManager
from src.logger import get_logger
from src.ml_engine import CLUSTER_ROLES, analysis_columns, feature_lists, train_classifiers, train_kmeans
from src.pipeline import run_cleaning_pipeline
from src.spark_engine import get_spark, read_parquet
from config import DISEASE_ROLES, ML_TARGET_ROLES

log = get_logger("run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--hdfs", action="store_true", help="read the HDFS copy and write the cleaned data to HDFS")
    ap.add_argument("--force", action="store_true", help="ignore cached cleaning output")
    args = ap.parse_args()

    src = Path(args.dataset)
    fp = file_fingerprint(src)
    storage = "LOCAL"
    source = str(src)
    if args.hdfs:
        mgr = HDFSManager()
        st_ = mgr.status()
        if not st_["connected"]:
            raise SystemExit(f"HDFS is not reachable ({st_['error']}). Start Hadoop first (see README).")
        mgr.ensure_dirs(True)
        mgr.upload(src, HDFS_INPUT_DIR, connected=True)
        source = f"{mgr.fs_uri}{HDFS_INPUT_DIR}/{src.name}"
        storage = "HDFS"

    spark = get_spark()
    t0 = time.time()
    stats = run_cleaning_pipeline(spark, source, fingerprint=fp, source_name=src.name, storage=storage,
                                  force=args.force,
                                  progress=lambda p, m: print(f"  [{int(p*100):3d}%] {m}"))
    print(f"cleaning: {stats['status']} · storage {storage} · {stats['cleaned_rows']:,} valid, "
          f"{stats['rejected_rows']:,} rejected, reconciles={stats['reconciles']}")
    cp = stats["cleaned_path"]

    print("Spark SQL (9 queries) ...")
    sql = sql_results(spark, cp)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    sql_out = {}
    for name, res in sql.items():
        if res.get("frame") is not None:
            res["frame"].to_csv(SUMMARY_DIR / f"{name}.csv", index=False)
            sql_out[name] = {"title": res["title"], "seconds": res["seconds"],
                             "rows": res["frame"].to_dict(orient="records")}
        else:
            sql_out[name] = {"title": res["title"], "error": res["error"]}

    roles = stats["roles_detected"]
    all_cols = list(stats["after"]["missing_by_column"].keys())
    columns = analysis_columns(roles, all_cols)
    print(f"Spark samples up to 50,000 records of {len(columns)} modelling columns for scikit-learn ...")
    pdf = ml_frame(spark, cp, columns)

    print("Machine learning (scikit-learn) ...")
    targets = [roles[r] for r in ML_TARGET_ROLES if r in roles]
    ml_out = {}
    for tgt in targets:
        num, cat = feature_lists(roles, columns, tgt, pdf)
        t = time.time()
        b = train_classifiers(pdf, tgt, num, cat)
        ml_out[tgt] = {
            "labels": b["labels"], "majority_baseline_accuracy": b["majority_baseline_accuracy"],
            "sample_rows": int(len(pdf)),
            "rows_used": b["rows_used"], "train_rows": b["train_rows"],
            "test_rows": b["test_rows"], "class_counts": b["class_counts"],
            "features_numeric": num, "features_categorical": cat, "seconds": round(time.time() - t, 2),
            "models": {n: {"metrics": r["metrics"], "confusion_matrix": r["confusion_matrix"],
                           "importance_top10": r["importance"].head(10).to_dict(orient="records"),
                           "fit_seconds": r["fit_seconds"]} for n, r in b["results"].items()},
        }
        m = {n: r["metrics"]["accuracy"] for n, r in b["results"].items()}
        print(f"  {tgt}: accuracy {m}")

    print("K-Means (exploratory) ...")
    feats = [roles[r] for r in CLUSTER_ROLES if r in roles and roles[r] in pdf.columns]
    km = train_kmeans(pdf, feats, k=3)
    km_out = {"k": km["k"], "features": km["features"], "rows_used": km["rows_used"],
              "silhouette": km["silhouette"], "profile": km["profile"].to_dict(orient="records")}

    print("Dashboard aggregates (Spark) ...")
    ov = overview(spark, cp)
    dv = disease_view(spark, cp)
    results = {
        "dataset": {"file": src.name, "size_bytes": stats["file_size_bytes"], "size_mb": stats["file_size_mb"],
                    "fingerprint": fp, "storage": storage, "source": source},
        "before": stats["before"], "after": stats["after"], "duplicates_removed": stats["duplicates_removed"],
        "rejected_rows": stats["rejected_rows"], "invalid_breakdown": stats["invalid_breakdown"],
        "consistency_warnings_kept": stats["consistency_warnings_kept"], "cleaned_rows": stats["cleaned_rows"],
        "roles_detected": roles, "codebook_changes": stats.get("codebook_changes"), "timings_s": stats["timings_s"], "cleaning_total_s": stats["total_s"],
        "spark": {k: v for k, v in stats["spark"].items() if k != "ui_url"},
        "sql": sql_out, "ml": ml_out, "kmeans": km_out,
        "overview_diseases": ov["diseases"].to_dict(orient="records") if ov["diseases"] is not None else None,
        "overview_age_groups": ov["age_groups"].to_dict(orient="records") if ov["age_groups"] is not None else None,
        "risk_distribution": ov["risk"].to_dict(orient="records") if ov["risk"] is not None else None,
        "disease_by_age": {k: v.to_dict(orient="records") for k, v in dv["by_age"].items()},
        "disease_by_gender": {k: v.to_dict(orient="records") for k, v in dv["by_gender"].items()},
        "clinical_summary": dv["clinical"].to_dict(orient="records") if dv["clinical"] is not None else None,
        "total_seconds": round(time.time() - t0, 2),
    }
    out = OUTPUT_DIR / f"run_results_{storage.lower()}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({round(time.time() - t0, 1)} s in total)")
    spark.stop()


if __name__ == "__main__":
    main()
