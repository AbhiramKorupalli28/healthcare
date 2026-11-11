"""Reports: CSV downloads, analytical summaries, visual reports and key findings."""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import REPORTS_DIR, SUMMARY_DIR
from src.app_state import cached_frame, ensure_cleaned
from src.ml_engine import analysis_columns
from src.ui import note, page_header, pretty


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _sql_frames(path: str) -> dict:
    return cached_frame("sql", path, "sql_results")


def main():
    page_header("Reports", "Downloads, analytical summaries, visual reports and key findings")
    stats = ensure_cleaned()
    path = stats["cleaned_path"]
    ov = cached_frame("overview", path, "overview")
    sql = _sql_frames(path)

    st.subheader("CSV downloads")
    # the full cleaned table has 1.3 million rows: only a Spark-sampled slice of the analysis
    # columns is offered for download (the whole file stays in the Parquet output)
    sample = cached_frame("sample", path, "ml_frame",
                          json.dumps({"columns": analysis_columns(stats["roles_detected"],
                                      list(stats["after"]["missing_by_column"].keys())), "max_rows": 5000}))
    st.download_button("Cleaned analysis columns, random sample of 5,000 valid records (CSV)",
                       _csv_bytes(sample), file_name="cleaned_sample_5000.csv", mime="text/csv")
    if ov["diseases"] is not None:
        st.download_button("Disease frequency (CSV)", _csv_bytes(ov["diseases"]),
                           file_name="disease_frequency.csv", mime="text/csv")

    st.subheader("Analytical summaries (Spark SQL)")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for name, res in sql.items():
        if res.get("frame") is None:
            note(f"{res['title']}: {res['error']}.")
            continue
        frame = res["frame"]
        out = SUMMARY_DIR / f"{name}.csv"
        frame.to_csv(out, index=False)
        with st.expander(f"{res['title']} · {len(frame)} rows · {res['seconds']} s"):
            st.dataframe(frame, use_container_width=True)
            st.download_button("Download CSV", _csv_bytes(frame), file_name=f"{name}.csv", mime="text/csv",
                               key=f"dl_{name}")

    st.subheader("Visual reports")
    if st.button("Generate visual report images"):
        from src.visualization import save_png_bar
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        made = []
        if ov["diseases"] is not None:
            made.append(save_png_bar(ov["diseases"], "condition", "positive", "Patients with each condition",
                                     REPORTS_DIR / "disease_distribution.png"))
        if ov["age_groups"] is not None:
            made.append(save_png_bar(ov["age_groups"].sort_values("age_group"), "age_group", "count",
                                     "Age distribution (valid records)", REPORTS_DIR / "age_distribution.png"))
        if ov["risk"] is not None:
            made.append(save_png_bar(ov["risk"], ov["risk"].columns[0], "count",
                                     "Risk categories (dataset label)", REPORTS_DIR / "risk_distribution.png"))
        st.session_state["reports_made"] = [str(m) for m in made]
    for p in st.session_state.get("reports_made", []):
        if Path(p).exists():
            st.image(p, caption=Path(p).name)
    if st.session_state.get("reports_made"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for p in st.session_state["reports_made"]:
                z.write(p, Path(p).name)
        st.download_button("Download visual report (ZIP of PNG files)", buf.getvalue(),
                           file_name="visual_report.zip", mime="application/zip")

    st.subheader("Key findings (computed from this run)")
    findings = _findings(stats, ov, sql)
    for f in findings:
        st.markdown(f"- {f}")


def _findings(stats, ov, sql) -> list[str]:
    out = []
    b, a = stats["before"], stats["after"]
    out.append(f"{stats['cleaned_rows']:,} of {b['rows']:,} records pass validation; "
               f"{stats['rejected_rows']:,} were rejected for systolic pressure not above diastolic, "
               f"and {b['duplicate_rows']} exact duplicates were removed.")
    d = ov["diseases"]
    if d is not None and len(d):
        top = d.iloc[0]
        out.append(f"Most common condition: {pretty(top['condition'])} at {top['prevalence_pct']:.1f}% "
                   f"({top['positive']:,} of {top['known']:,}).")
    r = ov["risk"]
    if r is not None and len(r):
        tot = r["count"].sum()
        parts = ", ".join(f"{row.iloc[0]} {100.0 * row['count'] / tot:.1f}%" for _, row in r.iterrows())
        out.append(f"Risk-label split (dataset's own label): {parts}.")
    bp = sql.get("7_blood_pressure", {}).get("frame")
    if bp is not None and len(bp):
        out.append("Blood-pressure classes (AHA 2017): " +
                   ", ".join(f"{row['bp_category']} {int(row['patients']):,}" for _, row in bp.iterrows()) + ".")
    out.append("Survey counts are unweighted sample counts of the CDC files; they describe respondents, "
               "not population rates. The models describe patterns in these answers, not diagnoses.")
    return out


main()
