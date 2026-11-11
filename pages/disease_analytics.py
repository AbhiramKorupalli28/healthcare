"""Disease Analytics: frequency, age-wise and gender-wise patterns and clinical measures."""
import json

import streamlit as st

from src import visualization as viz
from src.app_state import cached_frame, ensure_cleaned
from src.ui import note, page_header, pretty, source_note


def main():
    page_header("Disease Analytics", "Disease frequency, age and gender patterns and clinical measures. Filters apply to every chart.")
    source_note()
    stats = ensure_cleaned()
    path = stats["cleaned_path"]
    opts = cached_frame("opts", path, "filter_options")

    c1, c2 = st.columns(2)
    genders = c1.multiselect("Gender", opts["genders"], default=opts["genders"])
    ages = c2.multiselect("Age group", opts["age_groups"], default=opts["age_groups"])
    args = json.dumps({"genders": sorted(genders), "age_groups": sorted(ages)})
    dv = cached_frame("disease", path, "disease_view", args)
    st.caption(f"{dv['rows']:,} records match these filters (empty selection means all).")
    if dv["rows"] == 0:
        note("No records match these filters.")
        return

    freq = dv["freq"]
    flags = [] if freq is None else list(freq["condition"])
    tabs = st.tabs(["Disease frequency", "Age-wise", "Gender-wise", "Clinical measurements"])

    with tabs[0]:
        if freq is not None and len(freq):
            f = freq.assign(condition=freq["condition"].map(pretty))
            st.plotly_chart(viz.bar(f, "condition", "prevalence_pct", "Prevalence (% of records with a known answer)"),
                            use_container_width=True)
            st.dataframe(f[["condition", "positive", "known", "prevalence_pct"]], use_container_width=True)
        else:
            note("No Yes/No condition columns were detected.")

    with tabs[1]:
        if flags and dv["by_age"]:
            flag = st.selectbox("Condition", flags, format_func=pretty, key="age_flag")
            df = dv["by_age"][flag]
            st.plotly_chart(viz.bar(df, "age_group", "positive_pct", f"{pretty(flag)} by age group (%)"),
                            use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else:
            note("Age groups are not available in this dataset.")

    with tabs[2]:
        if flags and dv["by_gender"]:
            flag = st.selectbox("Condition ", flags, format_func=pretty, key="gender_flag")
            df = dv["by_gender"][flag]
            st.plotly_chart(viz.bar(df, list(df.columns)[0], "positive_pct", f"{pretty(flag)} by gender (%)"),
                            use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else:
            note("Gender is not available in this dataset.")

    with tabs[3]:
        if dv["clinical"] is not None and len(dv["clinical"]):
            st.markdown("**Measurements across the filtered records (Spark)**")
            st.dataframe(dv["clinical"].assign(measure=dv["clinical"]["measure"].map(pretty)),
                         use_container_width=True)
        if flags and dv["measures_by_flag"]:
            flag = st.selectbox("Average measurements by", flags, format_func=pretty, key="meas_flag")
            df = dv["measures_by_flag"][flag]
            st.markdown(f"**Average measurements by {pretty(flag)} status**")
            st.dataframe(df, use_container_width=True)
        else:
            note("No clinical measurement columns were detected.")


main()
