"""Big Data Processing: HDFS status and files, Spark status, processing statistics, storage and logs."""
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from config import CLEAN_DIR, HDFS_INPUT_DIR, HDFS_PROCESSED_DIR, LOCAL_HDFS_DIR, OUTPUT_DIR
from src.app_state import active_dataset_path, ensure_cleaned, spark_session
from src.hdfs_manager import HDFSManager
from src.logger import tail_log
from src.spark_engine import spark_info
from src.ui import note, page_header, storage_badge


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def main():
    page_header("Big Data Processing", "HDFS status, Spark status, processing statistics and storage")
    mgr = HDFSManager()
    st_ = mgr.status()
    mode = st_["mode"]

    st.markdown(f"**Storage in use:** {storage_badge(mode)}", unsafe_allow_html=True)
    if mode != "HDFS":
        note(f"HDFS is not reachable ({st_['error']}). Files and processing below are on the "
             f"LOCAL FALLBACK, which is not HDFS. Start the NameNode and DataNode to use HDFS (see the README).")
    c1, c2, c3 = st.columns(3)
    c1.metric("NameNode", "connected" if st_["connected"] else "not reachable")
    c2.metric("Live DataNodes", st_["live_datanodes"] if st_["live_datanodes"] is not None else "n/a")
    c3.metric("Checked", st_["checked_at"].split(" ")[1])
    st.caption(f"NameNode {st_['namenode_url']} · file system {st_['fs_uri']}")

    st.subheader("HDFS files")
    b1, b2 = st.columns(2)
    if b1.button("Create /healthcare/input and /healthcare/processed"):
        mgr.ensure_dirs(st_["connected"])
        st.success("Directories are in place.")
    if b2.button("Upload the active dataset to HDFS input"):
        try:
            where = mgr.upload(active_dataset_path(), HDFS_INPUT_DIR, connected=st_["connected"])
            st.success(f"Stored: {where}")
        except Exception as exc:  # show the real error, do not hide it
            st.error(f"Upload did not complete: {exc}")
    for d in (HDFS_INPUT_DIR, HDFS_PROCESSED_DIR):
        st.markdown(f"**{d}** ({mode})")
        rows = mgr.list_dir(d, st_["connected"])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            note(f"Nothing under {d} yet.")

    st.subheader("Spark")
    info = spark_info(spark_session())
    st.dataframe(pd.DataFrame([{"item": k, "value": str(v)} for k, v in info.items()]),
                 use_container_width=True, hide_index=True)
    note("Spark runs in local mode on this machine (all CPU cores). The same code runs on a cluster by changing "
         "SPARK_MASTER and the HDFS address in config.py.")

    st.subheader("Processing statistics")
    stats = ensure_cleaned()
    t = stats.get("timings_s", {})
    tm = pd.DataFrame([{"step": k.replace("_s", "").replace("_", " "), "seconds": v} for k, v in t.items()])
    if len(tm):
        st.dataframe(tm, use_container_width=True, hide_index=True)
    st.write(f"Status: **{stats.get('status')}** · storage: **{stats.get('storage_mode')}** · "
             f"reads from **{stats.get('source_storage')}** · total **{stats.get('total_s')} s** · "
             f"reconciles: **{stats.get('reconciles')}**")

    if st.button("Re-run the Spark cleaning from the HDFS copy"):
        if st_["connected"]:
            hdfs_src = f"{mgr.fs_uri}{HDFS_INPUT_DIR}/{active_dataset_path().name}"
            from src.app_state import active_fingerprint
            from src.pipeline import run_cleaning_pipeline
            res = run_cleaning_pipeline(spark_session(), hdfs_src, fingerprint=active_fingerprint(),
                                        source_name=active_dataset_path().name, storage="HDFS", force=True)
            st.success(f"Processed from HDFS in {res['total_s']} s; output {res['cleaned_path']}")
        else:
            st.error("HDFS is not reachable, so the run was not made on HDFS.")

    st.subheader("Storage information")
    rows = [{"location": "local cleaned output", "path": str(CLEAN_DIR), "size_mb": round(_dir_size(CLEAN_DIR) / 1e6, 2)},
            {"location": "local uploads and fallback", "path": str(LOCAL_HDFS_DIR), "size_mb": round(_dir_size(LOCAL_HDFS_DIR) / 1e6, 2)},
            {"location": "processing logs", "path": str(OUTPUT_DIR / 'logs'), "size_mb": round(_dir_size(OUTPUT_DIR / 'logs') / 1e6, 3)}]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    free = shutil.disk_usage(str(OUTPUT_DIR)).free / 1e9
    st.caption(f"Free disk space on this machine: {free:.1f} GB")

    st.subheader("Processing log")
    lines = tail_log(25)
    st.code("\n".join(lines) if lines else "No processing has run yet.", language="text")


main()
