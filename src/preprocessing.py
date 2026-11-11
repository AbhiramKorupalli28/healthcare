"""Cleaning and validation, written as distributed PySpark transformations.

Steps (each one counted, so before/after statistics can be shown):
  1. standardise column names (snake_case, unique)
  2. profile: rows, columns, missing cells, duplicate rows
  3. remove exact duplicate rows
  4. convert types (numeric roles to double, text trimmed, empty text -> null)
  5. flag invalid records: values outside plausible ranges, or systolic <= diastolic
     -> invalid records go to a separate 'rejected' output, not into the analysis
  6. flag consistency warnings that are kept (LDL + HDL above total cholesterol)
  7. add derived categories with documented thresholds (age group, WHO BMI class,
     AHA 2017 blood-pressure class)
Missing values are NOT imputed here; they stay null and are handled inside the
machine-learning pipeline, fitted on training data only.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType, ShortType

from config import VALID_RANGES
from src.data_loader import to_snake

NUMERIC_TYPES = (IntegerType, LongType, DoubleType, FloatType, ShortType)


def standardize_columns(df: DataFrame) -> tuple[DataFrame, dict]:
    """Rename every column to unique snake_case. Returns the frame and the old->new mapping."""
    seen, new_names = {}, []
    for c in df.columns:
        base = to_snake(c)
        n = seen.get(base, 0)
        seen[base] = n + 1
        new_names.append(base if n == 0 else f"{base}_{n}")
    mapping = dict(zip(df.columns, new_names))
    return df.toDF(*new_names), mapping


def missing_expr(c: str):
    """1 if a cell is null or blank text, else 0."""
    return F.when(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""), 1).otherwise(0)


def profile(df: DataFrame) -> dict:
    """Row, column, missing-cell and duplicate counts. Each is a Spark job over the data."""
    n = df.count()
    agg = df.select([F.sum(missing_expr(c)).alias(c) for c in df.columns]).first().asDict()
    missing_by_column = {c: int(agg[c] or 0) for c in df.columns}
    dup_rows = n - df.dropDuplicates().count()
    return {
        "rows": n,
        "columns": len(df.columns),
        "missing_cells": sum(missing_by_column.values()),
        "missing_by_column": missing_by_column,
        "duplicate_rows": dup_rows,
    }


def numeric_columns(df: DataFrame) -> list[str]:
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NUMERIC_TYPES)]


def convert_types(df: DataFrame, roles: dict) -> DataFrame:
    """Text columns: trimmed, blank text -> null. Numeric-looking roles: cast to double.

    Done in one select, so a wide table (hundreds of columns) builds a single plan.
    """
    numeric = set(numeric_columns(df))
    exprs = []
    for f in df.schema.fields:
        c = f.name
        if c not in numeric and f.dataType.typeName() == "string":
            col = F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c)))
            if any(roles.get(r) == c for r in VALID_RANGES):
                col = col.cast(DoubleType())
            exprs.append(col.alias(c))
        else:
            exprs.append(F.col(c))
    return df.select(*exprs)


def flag_invalid(df: DataFrame, roles: dict) -> DataFrame:
    """Add 'validation_issues' (text reasons, empty when clean) and 'is_valid'."""
    reasons = []
    for role, (lo, hi) in VALID_RANGES.items():
        if role in roles:
            c = roles[role]
            reasons.append(F.when((F.col(c) < lo) | (F.col(c) > hi), F.lit(f"{role}_out_of_range")))
    if "bp_systolic" in roles and "bp_diastolic" in roles:
        s, d = roles["bp_systolic"], roles["bp_diastolic"]
        reasons.append(F.when(F.col(s) <= F.col(d), F.lit("systolic_not_above_diastolic")))
    if not reasons:
        return df.withColumn("validation_issues", F.lit("")).withColumn("is_valid", F.lit(True))
    issues = F.concat_ws(";", *reasons)
    return (df.withColumn("validation_issues", issues)
              .withColumn("is_valid", F.col("validation_issues") == ""))


def consistency_warnings(df: DataFrame, roles: dict) -> dict:
    """Counts of cross-column checks that are reported but kept in the data."""
    out = {}
    if all(k in roles for k in ("ldl", "hdl", "cholesterol")):
        l, h, t = roles["ldl"], roles["hdl"], roles["cholesterol"]
        out["ldl_plus_hdl_above_total_cholesterol"] = df.filter(
            (F.col(l) + F.col(h)) > F.col(t) + 0.5).count()
    if all(k in roles for k in ("height_cm", "weight_kg", "bmi")):
        h, w, b = roles["height_cm"], roles["weight_kg"], roles["bmi"]
        calc = F.col(w) / ((F.col(h) / 100.0) ** 2)
        out["bmi_differs_from_height_weight_by_over_1"] = df.filter(
            F.abs(calc - F.col(b)) > 1.0).count()
    return out


def add_derived(df: DataFrame, roles: dict) -> DataFrame:
    """Documented derived categories. Each is added only when its inputs exist."""
    # A missing input stays missing: the first branch of every chain returns null for it.
    if "age" in roles:
        a = roles["age"]
        df = df.withColumn("age_group", F.when(F.col(a).isNull(), None)
                           .when(F.col(a) < 30, "18-29")
                           .when(F.col(a) < 40, "30-39").when(F.col(a) < 50, "40-49")
                           .when(F.col(a) < 60, "50-59").when(F.col(a) < 70, "60-69")
                           .otherwise("70+"))
    if "bmi" in roles:
        b = roles["bmi"]
        # WHO adult BMI classes
        df = df.withColumn("bmi_category", F.when(F.col(b).isNull(), None)
                           .when(F.col(b) < 18.5, "Underweight")
                           .when(F.col(b) < 25, "Normal")
                           .when(F.col(b) < 30, "Overweight").otherwise("Obese"))
    if "bp_systolic" in roles and "bp_diastolic" in roles:
        s, d = roles["bp_systolic"], roles["bp_diastolic"]
        # ACC/AHA 2017 blood-pressure classes (the higher of the two readings decides)
        df = df.withColumn("bp_category", F.when(F.col(s).isNull() | F.col(d).isNull(), None)
                           .when((F.col(s) >= 140) | (F.col(d) >= 90), "Stage 2 hypertension")
                           .when((F.col(s) >= 130) | (F.col(d) >= 80), "Stage 1 hypertension")
                           .when(F.col(s) >= 120, "Elevated")
                           .otherwise("Normal"))
    return df


def clean(df: DataFrame, roles: dict) -> tuple[DataFrame, DataFrame]:
    """Return (cleaned, rejected). Cleaned has no duplicates and only valid records."""
    df = df.dropDuplicates()
    df = convert_types(df, roles)
    df = flag_invalid(df, roles)
    rejected = df.filter(~F.col("is_valid")).drop("is_valid")
    cleaned = df.filter(F.col("is_valid")).drop("is_valid", "validation_issues")
    cleaned = add_derived(cleaned, roles)
    return cleaned, rejected


def invalid_breakdown(df: DataFrame) -> dict:
    """How many records each validation reason flagged (a record can have several)."""
    if "validation_issues" not in df.columns:
        return {}
    rows = (df.filter(F.col("validation_issues") != "")
              .select(F.explode(F.split("validation_issues", ";")).alias("reason"))
              .groupBy("reason").count().collect())
    return {r["reason"]: int(r["count"]) for r in rows}
