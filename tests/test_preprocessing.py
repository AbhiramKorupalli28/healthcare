from pyspark.sql import functions as F

from config import VALID_RANGES
from src.codebook import apply_codebook
from src.data_loader import detect_roles
from src.preprocessing import add_derived, clean, profile, standardize_columns


def _prepared(sample_df):
    """Codebook first (as the pipeline does), then standardised names."""
    coded, _ = apply_codebook(sample_df)
    std, mapping = standardize_columns(coded)
    return std, mapping


def test_codebook_labels_scales_and_cleans_codes(spark):
    df = spark.createDataFrame([(1, 3, 2700.0, 88, 77, 1)],
                               ["SEXVAR", "DIABETE4", "_BMI5", "PHYSHLTH", "SLEPTIM1", "_TOTINDA"])
    out, changed = apply_codebook(df)
    row = out.collect()[0].asDict()
    assert row["gender"] == "Male"
    assert row["diabetes"] == "No"
    assert abs(row["bmi"] - 27.0) < 1e-9          # two implied decimals
    assert row["physical_health_days"] == 0       # 88 = none
    assert row["sleep_hours"] is None             # 77 = don't know
    assert row["physically_active"] == "Yes"
    assert "bmi / 100" in changed["scaled"]


def test_standardize_columns_unique_snake_case(spark):
    df = spark.createDataFrame([(1, 2)], ["Blood Pressure", "blood_pressure"])
    std, mapping = standardize_columns(df)
    assert std.columns == ["blood_pressure", "blood_pressure_1"]
    assert mapping["Blood Pressure"] == "blood_pressure"


def test_profile_counts_on_real_sample(sample_df):
    std, _ = _prepared(sample_df)
    p = profile(std)
    assert p["rows"] == 2000
    assert p["columns"] == len(sample_df.columns)
    assert p["duplicate_rows"] >= 0
    assert p["missing_by_column"]["gender"] == 0


def test_clean_reconciles_on_real_sample(sample_df):
    std, _ = _prepared(sample_df)
    roles = detect_roles(std.columns)
    cleaned, rejected = clean(std, roles)
    dups = std.count() - std.dropDuplicates().count()
    assert cleaned.count() + rejected.count() == std.count() - dups


def test_invalid_physical_health_days_is_rejected(sample_df):
    std, _ = _prepared(sample_df)
    roles = detect_roles(std.columns)
    bad = std.limit(1).withColumn(roles["physical_health_days"], F.lit(float(VALID_RANGES["physical_health_days"][1] + 15)))
    cleaned, rejected = clean(bad, roles)
    assert cleaned.count() == 0 and rejected.count() == 1
    assert "physical_health_days_out_of_range" in rejected.collect()[0]["validation_issues"]


def test_out_of_range_bmi_is_flagged(sample_df):
    std, _ = _prepared(sample_df)
    roles = detect_roles(std.columns)
    bad = std.limit(1).withColumn(roles["bmi"], F.lit(float(VALID_RANGES["bmi"][1] + 50)))
    cleaned, rejected = clean(bad, roles)
    assert cleaned.count() == 0 and rejected.count() == 1
    assert "bmi_out_of_range" in rejected.collect()[0]["validation_issues"]


def test_derived_categories_use_documented_thresholds(spark):
    df = spark.createDataFrame([(25, 17.0, 110, 70), (45, 27.0, 125, 75), (65, 33.0, 150, 95), (75, 22.0, 118, 78)],
                               ["age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic"])
    roles = {"age": "age", "bmi": "bmi", "bp_systolic": "blood_pressure_systolic",
             "bp_diastolic": "blood_pressure_diastolic"}
    out = {r["age"]: r for r in add_derived(df, roles).collect()}
    assert out[25]["age_group"] == "18-29" and out[75]["age_group"] == "70+"
    assert out[25]["bmi_category"] == "Underweight"
    assert out[45]["bmi_category"] == "Overweight"
    assert out[25]["bp_category"] == "Normal"
    assert out[45]["bp_category"] == "Elevated"
    assert out[65]["bp_category"] == "Stage 2 hypertension"
    assert out[75]["bp_category"] == "Normal"
    # a missing input must stay missing, not fall into the last class
    miss = spark.createDataFrame([(None, None, None, None)],
                                 "age int, bmi double, blood_pressure_systolic int, blood_pressure_diastolic int")
    r = add_derived(miss, roles).collect()[0].asDict()
    assert r["age_group"] is None and r["bmi_category"] is None and r["bp_category"] is None
