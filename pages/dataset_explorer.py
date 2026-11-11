"""Dataset Explorer: CSV upload, preview, schema, missing values, duplicates and column filter."""
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from config import UPLOAD_DIR
from src.app_state import active_dataset_path, cached_frame, ensure_cleaned, set_active_dataset
from src.data_loader import preview_pandas
from src.ui import metric_card, note, page_header


def _save_upload(upload) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(upload.name).name)
    dest = UPLOAD_DIR / safe
    dest.write_bytes(upload.getbuffer())
    return dest


def main():
    page_header("Dataset Explorer", "Upload a CSV, then check its preview, schema, missing values and duplicates")
    up = st.file_uploader("Upload a healthcare CSV (large files are processed by Spark)", type=["csv"])
    if up is not None:
        token = f"{up.name}:{up.size}"
        if st.session_state.get("last_upload") != token:
            dest = _save_upload(up)
            set_active_dataset(dest)
            st.session_state["last_upload"] = token
            st.rerun()

    path = active_dataset_path()
    stats = ensure_cleaned()
    before = stats["before"]

    c = st.columns(4)
    with c[0]:
        metric_card("File", path.name, f"{stats.get('file_size_mb') or 0:.2f} MB")
    with c[1]:
        metric_card("Rows (raw)", f"{before['rows']:,}", f"{stats['cleaned_rows']:,} valid after cleaning")
    with c[2]:
        metric_card("Columns", f"{before['columns']}", f"{stats['raw_columns']} in the file header")
    with c[3]:
        metric_card("Duplicate rows", f"{before['duplicate_rows']:,}", f"{before['missing_cells']:,} missing cells")

    st.subheader("Preview")
    st.caption("First 100 records, read for display. All analysis uses Spark over the whole file.")
    prev = preview_pandas(path, 100)
    cols = st.multiselect("Column filter", list(prev.columns), default=list(prev.columns))
    st.dataframe(prev[cols] if cols else prev, use_container_width=True, height=320)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Schema (Spark)")
        sch = cached_frame("schema", str(path), "raw_schema", "{}")
        schema_df = pd.DataFrame(sch["schema"])
        if cols:
            schema_df = schema_df[schema_df["column"].isin(cols)]
        st.dataframe(schema_df, use_container_width=True, height=380)
    with right:
        st.subheader("Missing values")
        miss = pd.DataFrame({"column": list(before["missing_by_column"].keys()),
                             "missing": list(before["missing_by_column"].values())})
        miss["missing_pct"] = (100.0 * miss["missing"] / before["rows"]).round(2)
        miss = miss[miss["missing"] > 0].sort_values("missing", ascending=False)
        if cols:
            miss = miss[miss["column"].isin(cols)]
        if len(miss):
            st.dataframe(miss, use_container_width=True, height=380)
        else:
            note("No missing values in the selected columns.")
        st.subheader("Duplicates")
        note(f"{before['duplicate_rows']:,} exact duplicate rows in the raw file "
             f"({stats['duplicates_removed']:,} removed in cleaning).")


main()
