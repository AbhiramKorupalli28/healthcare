"""Risk Analysis: the dataset's own risk labels, scikit-learn models, K-Means and a prediction form.

Outputs describe patterns in the survey file. They are not medical diagnoses.
"""
import json

import pandas as pd
import streamlit as st

from config import ML_TARGET_ROLES
from src import visualization as viz
from src.app_state import cached_frame, ensure_cleaned, spark_session
from src.ml_engine import (CLUSTER_ROLES, DISCLAIMER, analysis_columns, feature_lists, form_defaults,
                           predict_one, train_classifiers, train_kmeans)
from src.ui import note, page_header, pretty, source_note


@st.cache_resource(show_spinner="Training Decision Tree and Random Forest (scikit-learn)...")
def model_bundle(path: str, target: str, columns: tuple, roles_json: str):
    pdf = cached_frame("ml", path, "ml_frame", json.dumps({"columns": list(columns)})).copy()
    num, cat = feature_lists(json.loads(roles_json), list(columns), target, pdf)
    return train_classifiers(pdf, target, num, cat) | {"pdf": pdf}


def main():
    page_header("Risk Analysis", "Risk categories from the dataset's own labels, and model results")
    source_note()
    stats = ensure_cleaned()
    path = stats["cleaned_path"]
    roles = stats["roles_detected"]
    columns = tuple(analysis_columns(roles, list(stats["after"]["missing_by_column"].keys())))

    st.subheader("Risk category distribution")
    rc = cached_frame("risk", path, "risk_counts")
    if rc["risk"] is not None and len(rc["risk"]):
        r = rc["risk"]
        r = r.assign(pct=(100.0 * r["count"] / r["count"].sum()).round(2))
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(viz.bar(r, r.columns[0], "count", "Patients per risk class (dataset label)"),
                            use_container_width=True)
        with right:
            st.dataframe(r, use_container_width=True)
            note("This is the CDC-coded BMI category (computed by CDC from measured or reported height and weight). "
                 "It is used here as the risk category.")
    else:
        note("This dataset has no risk-category column, so no risk distribution is shown.")

    st.subheader("Machine learning")
    targets = []
    if "risk_label" in roles:
        targets.append(roles["risk_label"])
    targets += [roles[r] for r in ML_TARGET_ROLES[1:] if r in roles]
    if not targets:
        note("No target column with validated labels was found. Clustering is shown below instead.")
    else:
        target = st.selectbox("Target to predict (existing label in the dataset)", targets, format_func=pretty)
        st.caption("Targets are only the dataset's own columns. Nothing is invented or relabelled here. "
                   "Borderline answers are excluded from yes/no targets.")
        if st.button("Train models on the cleaned data", type="primary"):
            st.session_state[f"trained::{target}"] = True
        if st.session_state.get(f"trained::{target}"):
            bundle = model_bundle(path, target, columns, json.dumps(roles))
            _show_models(bundle)
            _predict_form(bundle)

    st.subheader("Exploratory clustering (K-Means, no labels)")
    pdf_cols = [roles[r] for r in CLUSTER_ROLES if r in roles]
    if len(pdf_cols) >= 2:
        k = st.slider("Number of clusters", 2, 6, 3)
        pdf = cached_frame("ml", path, "ml_frame", json.dumps({"columns": list(columns)}))
        km = _kmeans(pdf, tuple(pdf_cols), k)
        st.dataframe(km["profile"], use_container_width=True)
        st.caption(f"{km['rows_used']:,} records used (rows with a missing value in these measures are left out). "
                   f"Silhouette score {km['silhouette']} on a 5,000-record sample.")
        note(km["limits"])
    else:
        note("Too few numeric clinical measures were detected to cluster.")


@st.cache_data(show_spinner="Clustering with K-Means...")
def _kmeans(pdf, features, k):
    return train_kmeans(pdf, list(features), k=k)


def _show_models(bundle):
    st.caption(f"{bundle['rows_used']:,} records (up to the configured cap), "
               f"{bundle['train_rows']:,} for training and {bundle['test_rows']:,} for testing "
               f"(stratified split). Class counts: {bundle['class_counts']}.")
    rows = []
    for name, r in bundle["results"].items():
        m = r["metrics"]
        row = {"model": name, "accuracy": m["accuracy"], "precision (weighted)": m["precision_weighted"],
               "recall (weighted)": m["recall_weighted"], "F1 (weighted)": m["f1_weighted"],
               "F1 (macro)": m["f1_macro"]}
        if "f1_positive" in m:
            row.update({"precision (positive)": m["precision_positive"], "recall (positive)": m["recall_positive"],
                        "F1 (positive)": m["f1_positive"]})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)

    model = st.radio("Model to inspect", list(bundle["results"]), horizontal=True)
    r = bundle["results"][model]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.heatmap(r["confusion_matrix"], bundle["labels"], f"Confusion matrix: {model}"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(viz.importance_bar(r["importance"], f"Feature importance: {model}"),
                        use_container_width=True)
    with st.expander("Per-class report"):
        rep = pd.DataFrame(r["classification_report"]).T
        st.dataframe(rep.round(4), use_container_width=True)
    st.caption(bundle["disclaimer"])


def _predict_form(bundle):
    st.subheader("Prediction interface")
    note(DISCLAIMER)
    pdf = bundle["pdf"]
    num, cat = bundle["features_numeric"], bundle["features_categorical"]
    defaults = form_defaults(pdf, num, cat)
    with st.form("predict"):
        values = {}
        cols = st.columns(3)
        for i, c in enumerate(num):
            lo, hi = float(pdf[c].min()), float(pdf[c].max())
            values[c] = cols[i % 3].number_input(pretty(c), min_value=lo, max_value=hi,
                                                 value=float(defaults[c]), key=f"num_{c}")
        for i, c in enumerate(cat):
            options = sorted(pdf[c].dropna().astype(str).unique().tolist())
            idx = options.index(defaults[c]) if defaults[c] in options else 0
            values[c] = cols[i % 3].selectbox(pretty(c), options, index=idx, key=f"cat_{c}")
        go = st.form_submit_button("Predict with Random Forest")
    if go:
        out = predict_one(bundle, "Random Forest", values)
        st.success(f"Predicted class: **{out['prediction']}**")
        st.dataframe(pd.DataFrame({"class": list(out["probabilities"]),
                                   "probability": list(out["probabilities"].values())}), use_container_width=True)
        st.caption(out["disclaimer"])


main()
