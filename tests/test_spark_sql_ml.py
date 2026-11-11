import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from config import DISEASE_ROLES
from src.analytics import overview
from src.data_loader import detect_roles
from src.ml_engine import feature_lists, form_defaults, predict_one, train_classifiers, train_kmeans
from src.codebook import apply_codebook
from src.preprocessing import clean, standardize_columns
from src.spark_engine import clinical_summary, disease_frequency, rate_by
from src.sql_analytics import SparkSQLAnalytics
from tests.conftest import SAMPLE


def _cleaned(spark, sample_df):
    coded, _ = apply_codebook(sample_df)
    std, _ = standardize_columns(coded)
    roles = detect_roles(std.columns)
    cleaned, rejected = clean(std, roles)
    return cleaned, roles, rejected


def test_spark_disease_frequency_matches_pandas(spark, sample_df):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    flags = [roles[r] for r in DISEASE_ROLES if r in roles]
    assert "diabetes" in flags
    got = disease_frequency(cleaned, flags).set_index("condition")
    pdf = cleaned.toPandas()
    for f in flags:
        assert got.loc[f, "positive"] == int((pdf[f].str.upper() == "YES").sum())
        assert got.loc[f, "known"] == int(pdf[f].str.upper().isin(["YES", "NO"]).sum())


def test_spark_sql_total_and_risk_distribution(spark, sample_df):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    cleaned.createOrReplaceTempView("healthcare_patients")
    sql = SparkSQLAnalytics(spark, cleaned, roles)
    res = sql.run_all()
    n = len(cleaned.toPandas())
    assert res["1_total_records"]["frame"].iloc[0, 0] == n
    risk = res["9_risk_category"]["frame"]
    assert risk["patients"].sum() == n
    assert res["8_missing_values"]["frame"]["total"].iloc[0] == n
    assert len(res["8_missing_values"]["frame"]) == len(cleaned.columns)
    # the CDC survey has no glucose measurement, so that query is reported as unavailable, not guessed
    assert res["6_avg_glucose_by_disease"]["error"] is not None
    # blood-pressure readings and the hypertension question are not in all three survey years
    assert res["7_blood_pressure"]["error"] is not None


def test_sql_average_bmi_matches_pandas(spark, sample_df):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    sql = SparkSQLAnalytics(spark, cleaned, roles)
    frame = sql.run_all()["5_avg_bmi_by_disease"]["frame"]
    pdf = cleaned.toPandas()
    row = frame[(frame["condition"] == "diabetes") & (frame["status"] == "Yes")].iloc[0]
    expected = pdf.loc[pdf["diabetes"] == "Yes", roles["bmi"]].mean()
    assert abs(row["avg_bmi"] - expected) < 0.01


def test_rate_by_and_clinical_summary(spark, sample_df):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    rb = rate_by(cleaned, "gender", "diabetes")
    known = cleaned.filter(F.col("diabetes").isin("Yes", "No")).count()   # rate_by counts known answers only
    assert rb["patients"].sum() == known
    cs = clinical_summary(cleaned, [roles["bmi"]])
    assert cs.loc[0, "count"] == cleaned.filter(F.col(roles["bmi"]).isNotNull()).count()   # non-null values only


def test_train_classifiers_on_real_sample(spark, sample_df):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    pdf = cleaned.toPandas()
    cols = list(pdf.columns)
    target = roles["risk_label"]
    num, cat = feature_lists(roles, cols, target, pdf)
    assert "risk_score" not in num + cat and target not in num + cat
    assert roles["risk_label"] not in num + cat
    bundle = train_classifiers(pdf, target, num, cat, max_rows=2000, trees=30)
    for name, r in bundle["results"].items():
        cm = np.array(r["confusion_matrix"])
        assert cm.sum() == bundle["test_rows"]
        assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
        assert {"precision_weighted", "recall_weighted", "f1_weighted", "f1_macro"} <= set(r["metrics"])
    rf = bundle["results"]["Random Forest"]["importance"]
    assert set(rf["feature"]) == set(num + cat)
    out = predict_one(bundle, "Random Forest", form_defaults(pdf, num, cat))
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3
    assert "Not a medical diagnosis" in out["disclaimer"]


def test_kmeans_sizes_add_up(spark, sample_df):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    pdf = cleaned.toPandas()
    feats = [roles[r] for r in ["age", "bmi", "physical_health_days", "mental_health_days"]]
    km = train_kmeans(pdf, feats, k=3)
    assert int(km["profile"]["patients"].sum()) == km["rows_used"]
    assert -1.0 <= km["silhouette"] <= 1.0


def test_overview_counts_on_cleaned(spark, sample_df, tmp_path):
    cleaned, roles, _ = _cleaned(spark, sample_df)
    p = str(tmp_path / "cleaned.parquet")
    cleaned.write.mode("overwrite").parquet(p)
    ov = overview(spark, p)
    assert ov["rows"] == cleaned.count()
    assert int(ov["risk"]["count"].sum()) == ov["rows"]
