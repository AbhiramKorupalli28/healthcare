"""Page-level analyses. Each function reads the cleaned Parquet with Spark, runs Spark
aggregations, and returns small pandas tables for display. They are called through
src.app_state.cached_frame so each result is computed once per dataset version.
"""
from src.data_loader import detect_roles
from src.preprocessing import numeric_columns
from src.spark_engine import (clinical_summary, disease_frequency, filter_frame, mean_by, rate_by,
                              read_parquet, value_counts)
from src.sql_analytics import SparkSQLAnalytics
from config import DISEASE_ROLES


def _load(spark, path):
    df = read_parquet(spark, path)
    roles = detect_roles(df.columns)
    flags = [roles[r] for r in DISEASE_ROLES if r in roles]
    return df, roles, flags


def overview(spark, path: str) -> dict:
    df, roles, flags = _load(spark, path)
    out = {"rows": df.count(), "columns": len(df.columns),
           "diseases": disease_frequency(df, flags) if flags else None,
           "age_groups": value_counts(df, "age_group") if "age_group" in df.columns else None,
           "risk": value_counts(df, roles["risk_label"]) if "risk_label" in roles else None}
    return out


def disease_view(spark, path: str, genders: list | None = None, age_groups: list | None = None) -> dict:
    df, roles, flags = _load(spark, path)
    f = filter_frame(df, roles.get("gender"), genders, "age_group" if "age_group" in df.columns else None, age_groups)
    measures = [roles[r] for r in ["bmi", "glucose", "hba1c", "bp_systolic", "bp_diastolic", "cholesterol"] if r in roles]
    out = {"rows": f.count(), "freq": disease_frequency(f, flags) if flags else None,
           "clinical": clinical_summary(f, measures) if measures else None,
           "by_age": {}, "by_gender": {}, "measures_by_flag": {}}
    for fl in flags:
        if "age_group" in df.columns:
            out["by_age"][fl] = rate_by(f, "age_group", fl)
        if roles.get("gender"):
            out["by_gender"][fl] = rate_by(f, roles["gender"], fl)
        if measures:
            out["measures_by_flag"][fl] = mean_by(f.filter(f[fl].isNotNull()), fl, measures)
    out["measures"] = measures
    return out


def risk_counts(spark, path: str) -> dict:
    df, roles, _ = _load(spark, path)
    if "risk_label" not in roles:
        return {"risk": None}
    return {"risk": value_counts(df, roles["risk_label"])}


def sql_results(spark, path: str) -> dict:
    df, roles, _ = _load(spark, path)
    sql = SparkSQLAnalytics(spark, df, roles)
    res = sql.run_all()
    return {k: {kk: vv for kk, vv in v.items() if kk != "frame"} | {"frame": v["frame"]} for k, v in res.items()}


def numeric_list(spark, path: str) -> list:
    return numeric_columns(read_parquet(spark, path))


def filter_options(spark, path: str) -> dict:
    """Distinct gender values and age groups present in the cleaned data (filter choices)."""
    df, roles, _ = _load(spark, path)
    out = {"genders": [], "age_groups": []}
    if roles.get("gender"):
        out["genders"] = sorted(r[0] for r in df.select(roles["gender"]).distinct().collect() if r[0] is not None)
    if "age_group" in df.columns:
        out["age_groups"] = sorted(r[0] for r in df.select("age_group").distinct().collect() if r[0] is not None)
    return out


def raw_schema(spark, source_path: str) -> dict:
    """Schema of the ORIGINAL CSV as Spark infers it (name, type, nullable) and its row count."""
    from src.spark_engine import read_csv
    df = read_csv(spark, source_path)
    return {"rows": df.count(), "columns": len(df.columns),
            "schema": [{"column": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable}
                       for f in df.schema.fields]}


def ml_frame(spark, path: str, columns: list, max_rows: int | None = None) -> "pandas.DataFrame":
    """The modelling columns for a random sample of at most max_rows records, collected to pandas.

    Spark does the sampling and the column selection; only the sample reaches pandas.
    """
    from config import ML_MAX_ROWS, RANDOM_STATE
    max_rows = max_rows or ML_MAX_ROWS
    df = read_parquet(spark, path)
    keep = [c for c in columns if c in df.columns]
    df = df.select(*keep)
    n = df.count()
    if n > max_rows:
        df = df.sample(withReplacement=False, fraction=min(1.0, 1.2 * max_rows / n), seed=RANDOM_STATE).limit(max_rows)
    return df.toPandas()


def cleaned_frame(spark, path: str) -> "pandas.DataFrame":
    """The whole cleaned table as pandas (used for the CSV download only)."""
    return read_parquet(spark, path).toPandas()
