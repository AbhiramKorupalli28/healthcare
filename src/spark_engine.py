"""Spark session and distributed analytics helpers.

Every aggregation here is a Spark job over the full DataFrame. Only the small
aggregated result (a few rows) is brought back as pandas for display.
"""
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config import SPARK_DRIVER_MEMORY, SPARK_MASTER, SPARK_SHUFFLE_PARTITIONS
from src.preprocessing import numeric_columns

YES_VALUES = ["YES", "Y", "TRUE", "1", "1.0"]
NO_VALUES = ["NO", "N", "FALSE", "0", "0.0"]


def get_spark(app_name: str = "HealthcareAnalytics", master: str | None = None) -> SparkSession:
    """Create (or reuse) the local SparkSession used by the app and the tests."""
    spark = (SparkSession.builder.appName(app_name)
             .master(master or SPARK_MASTER)
             .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
             .config("spark.sql.shuffle.partitions", str(SPARK_SHUFFLE_PARTITIONS))
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def spark_info(spark: SparkSession) -> dict:
    sc = spark.sparkContext
    return {
        "version": spark.version,
        "master": sc.master,
        "app_name": sc.appName,
        "app_id": sc.applicationId,
        "default_parallelism": sc.defaultParallelism,
        "driver_memory": spark.conf.get("spark.driver.memory", "default"),
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "ui_url": sc.uiWebUrl,
    }


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    """CSV with a header row. Works for local paths and hdfs:// URIs."""
    return spark.read.option("header", True).option("inferSchema", True).csv(path)


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path)


def is_yes(c: str):
    return F.upper(F.trim(F.col(c).cast("string"))).isin(YES_VALUES)


def is_known(c: str):
    return F.upper(F.trim(F.col(c).cast("string"))).isin(YES_VALUES + NO_VALUES)


def filter_frame(df: DataFrame, gender_col=None, genders=None, age_group_col=None, age_groups=None) -> DataFrame:
    if gender_col and genders:
        df = df.filter(F.col(gender_col).isin(list(genders)))
    if age_group_col and age_groups:
        df = df.filter(F.col(age_group_col).isin(list(age_groups)))
    return df


def disease_frequency(df: DataFrame, flag_cols: list[str]) -> "pandas.DataFrame":
    """Positive count and prevalence for each Yes/No flag, in one Spark aggregation."""
    if not flag_cols:
        import pandas as pd
        return pd.DataFrame(columns=["condition", "positive", "known", "prevalence_pct"])
    exprs = []
    for c in flag_cols:
        exprs.append(F.sum(F.when(is_yes(c), 1).otherwise(0)).alias(f"{c}__yes"))
        exprs.append(F.sum(F.when(is_known(c), 1).otherwise(0)).alias(f"{c}__known"))
    row = df.agg(*exprs).first().asDict()
    out = []
    for c in flag_cols:
        yes, known = int(row[f"{c}__yes"] or 0), int(row[f"{c}__known"] or 0)
        out.append({"condition": c, "positive": yes, "known": known,
                    "prevalence_pct": round(100.0 * yes / known, 2) if known else None})
    import pandas as pd
    return pd.DataFrame(out).sort_values("positive", ascending=False).reset_index(drop=True)


def rate_by(df: DataFrame, group_col: str, flag_col: str) -> "pandas.DataFrame":
    """For each group: patients, positives of the flag, and the positive percentage."""
    rows = (df.filter(is_known(flag_col))
              .groupBy(group_col)
              .agg(F.count("*").alias("patients"),
                   F.sum(F.when(is_yes(flag_col), 1).otherwise(0)).alias("positive"))
              .toPandas())
    rows["positive_pct"] = (100.0 * rows["positive"] / rows["patients"]).round(2)
    return rows.sort_values(group_col).reset_index(drop=True)


def mean_by(df: DataFrame, by_col: str, value_cols: list[str]) -> "pandas.DataFrame":
    """Average of each numeric column for each value of by_col."""
    aggs = [F.count("*").alias("patients")] + [F.round(F.avg(c), 2).alias(f"avg_{c}") for c in value_cols]
    return df.groupBy(by_col).agg(*aggs).toPandas().sort_values(by_col).reset_index(drop=True)


def clinical_summary(df: DataFrame, value_cols: list[str]) -> "pandas.DataFrame":
    """count, mean, std, min, max for each numeric column (one Spark aggregation)."""
    import pandas as pd
    exprs = []
    for c in value_cols:
        exprs += [F.count(c).alias(f"{c}__n"), F.avg(c).alias(f"{c}__mean"),
                  F.stddev(c).alias(f"{c}__std"), F.min(c).alias(f"{c}__min"), F.max(c).alias(f"{c}__max")]
    row = df.agg(*exprs).first().asDict()
    out = []
    for c in value_cols:
        out.append({"measure": c,
                    "count": int(row[f"{c}__n"] or 0),
                    "mean": _r(row[f"{c}__mean"]), "std": _r(row[f"{c}__std"]),
                    "min": _r(row[f"{c}__min"]), "max": _r(row[f"{c}__max"])})
    return pd.DataFrame(out)


def value_counts(df: DataFrame, col: str) -> "pandas.DataFrame":
    """Count of each value; a missing value is shown as 'Missing' rather than a blank row."""
    labelled = df.withColumn(col, F.coalesce(F.col(col).cast("string"), F.lit("Missing")))
    return labelled.groupBy(col).count().orderBy(F.desc("count")).toPandas()


def export_parquet(df: DataFrame, path: str) -> None:
    df.write.mode("overwrite").parquet(path)


def timed(fn, *args, **kwargs):
    t = time.time()
    out = fn(*args, **kwargs)
    return out, round(time.time() - t, 3)


def _r(x):
    return None if x is None else round(float(x), 3)


def numeric_cols_of(df: DataFrame) -> list[str]:
    return numeric_columns(df)
