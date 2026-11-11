"""Overview: totals, data-quality indicators, disease and age distributions, summary."""
import streamlit as st

from src import visualization as viz
from src.app_state import cached_frame, ensure_cleaned
from src.ui import metric_card, note, page_header, pretty, source_note


def main():
    page_header("Overview", "Totals, data quality and distributions, all computed from the cleaned records")
    source_note()
    stats = ensure_cleaned()
    path = stats["cleaned_path"]
    ov = cached_frame("overview", path, "overview")
    before, after = stats["before"], stats["after"]
    diseases, risk, ages = ov["diseases"], ov["risk"], ov["age_groups"]

    raw_cells = before["rows"] * before["columns"]
    miss_pct = 100.0 * before["missing_cells"] / raw_cells if raw_cells else 0.0
    n_conditions = 0 if diseases is None else len(diseases)
    n_risk = 0 if risk is None else len(risk)

    c = st.columns(4)
    with c[0]:
        metric_card("Valid patient records", f"{stats['cleaned_rows']:,}",
                    f"of {before['rows']:,} raw rows")
    with c[1]:
        metric_card("Disease categories", f"{n_conditions + n_risk}",
                    f"{n_conditions} conditions tracked · {n_risk} risk classes")
    with c[2]:
        size = stats.get("file_size_mb")
        metric_card("Dataset size", f"{size} MB" if size else "n/a",
                    f"{before['columns']} columns, including survey_year")
    with c[3]:
        metric_card("Missing cells", f"{miss_pct:.2f}%",
                    f"{before['duplicate_rows']} duplicates · {stats['rejected_rows']} invalid rejected")

    st.write("")
    left, right = st.columns(2)
    with left:
        if diseases is not None and len(diseases):
            d = diseases.assign(condition=diseases["condition"].map(pretty))
            st.plotly_chart(viz.bar(d, "condition", "positive", "Patients with each condition"),
                            use_container_width=True)
        else:
            note("No Yes/No condition columns were detected in this dataset.")
    with right:
        if ages is not None and len(ages):
            a = ages.sort_values("age_group")
            st.plotly_chart(viz.bar(a, "age_group", "count", "Age distribution (valid records)"),
                            use_container_width=True)
        else:
            note("No age column was detected in this dataset.")

    if risk is not None and len(risk):
        st.plotly_chart(viz.bar(risk, "diabetes_risk" if "diabetes_risk" in risk else risk.columns[0],
                                "count", "Risk categories (the dataset's own label)"),
                        use_container_width=True)

    st.subheader("Summary")
    lines = [f"{stats['cleaned_rows']:,} valid records are analysed. "
             f"{before['duplicate_rows']} duplicate rows were removed and "
             f"{stats['rejected_rows']:,} records were rejected because systolic pressure was not above diastolic."]
    if diseases is not None and len(diseases):
        top = diseases.iloc[0]
        lines.append(f"The most common condition is {pretty(top['condition'])}: "
                     f"{top['positive']:,} of {top['known']:,} records ({top['prevalence_pct']:.1f}%).")
    if risk is not None and len(risk):
        r0 = risk.iloc[0]
        total = int(risk["count"].sum())
        lines.append(f"The largest risk class is {r0.iloc[0]} with {int(r0['count']):,} records "
                     f"({100.0 * r0['count'] / total:.1f}%).")
    lines.append(f"Missing values are kept as they are in the cleaned data; {after['missing_cells']:,} "
                 f"cells are still empty after cleaning, and the models impute them.")
    for ln in lines:
        st.markdown(f"- {ln}")


main()
