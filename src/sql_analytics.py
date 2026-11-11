"""Spark SQL analytics over a temporary view.

The nine queries from the brief are built from the detected column roles, so a
query whose columns are missing from a dataset is reported as unavailable
instead of failing or inventing a result.
"""
import time

from pyspark.sql import DataFrame, SparkSession

from config import DISEASE_ROLES
from src.logger import get_logger

log = get_logger("sql")
VIEW = "healthcare_patients"


class SparkSQLAnalytics:
    def __init__(self, spark: SparkSession, df: DataFrame, roles: dict):
        self.spark = spark
        self.roles = roles
        self.df = df
        df.createOrReplaceTempView(VIEW)
        self.cols = set(df.columns)

    # column helpers (return the standardised column name or None)
    def _c(self, role):
        c = self.roles.get(role)
        return c if c in self.cols else None

    def _diseases(self):
        return [self.roles[r] for r in DISEASE_ROLES if r in self.roles]

    def queries(self) -> dict:
        """name -> (description, sql or None with the reason it cannot run)."""
        q = {}
        q["1_total_records"] = ("Total patient records", f"SELECT COUNT(*) AS total_patient_records FROM {VIEW}")

        dis = self._diseases()
        if dis:
            parts = [f"SELECT '{c}' AS condition, SUM(CASE WHEN UPPER({c})='YES' THEN 1 ELSE 0 END) AS positive, "
                     f"SUM(CASE WHEN UPPER({c}) IN ('YES','NO') THEN 1 ELSE 0 END) AS known FROM {VIEW}" for c in dis]
            q["2_disease_counts"] = ("Disease-wise patient count", " UNION ALL ".join(parts))
        else:
            q["2_disease_counts"] = ("Disease-wise patient count", None)

        g = self._c("gender")
        age_group = "age_group" if "age_group" in self.cols else None
        if age_group and dis:
            pct = ", ".join(f"ROUND(100.0*AVG(CASE WHEN UPPER({c})='YES' THEN 1.0 ELSE 0.0 END),2) AS {c}_pct"
                            for c in dis)
            q["3_age_group_distribution"] = ("Age-group disease distribution",
                f"SELECT age_group, COUNT(*) AS patients, {pct} FROM {VIEW} GROUP BY age_group ORDER BY age_group")
        else:
            q["3_age_group_distribution"] = ("Age-group disease distribution", None)

        if g and dis:
            pct = ", ".join(f"ROUND(100.0*AVG(CASE WHEN UPPER({c})='YES' THEN 1.0 ELSE 0.0 END),2) AS {c}_pct"
                            for c in dis)
            q["4_gender_distribution"] = ("Gender-wise disease distribution",
                f"SELECT {g} AS gender, COUNT(*) AS patients, {pct} FROM {VIEW} GROUP BY {g} ORDER BY gender")
        else:
            q["4_gender_distribution"] = ("Gender-wise disease distribution", None)

        def avg_by_flag(role, label):
            col, flags = self._c(role), dis
            if not col or not flags:
                return (label, None)
            parts = [f"SELECT '{f}' AS condition, {f} AS status, COUNT(*) AS patients, "
                     f"ROUND(AVG({col}),2) AS avg_{role} FROM {VIEW} WHERE {f} IS NOT NULL GROUP BY {f}"
                     for f in flags]
            return (label, " UNION ALL ".join(parts))

        q["5_avg_bmi_by_disease"] = avg_by_flag("bmi", "Average BMI by disease")
        q["6_avg_glucose_by_disease"] = avg_by_flag("glucose", "Average glucose by disease")

        bp = self._c("bp_systolic"), self._c("bp_diastolic")
        if "bp_category" in self.cols and all(bp):
            q["7_blood_pressure"] = ("Blood pressure classes (AHA 2017)",
                f"SELECT bp_category, COUNT(*) AS patients, ROUND(AVG({bp[0]}),1) AS avg_systolic, "
                f"ROUND(AVG({bp[1]}),1) AS avg_diastolic FROM {VIEW} GROUP BY bp_category ORDER BY patients DESC")
        elif self._c("risk_label") and "hypertension" in self.roles:
            # survey file: blood pressure is a yes/no answer, so report it against BMI class
            r, h = self._c("risk_label"), self.roles["hypertension"]
            q["7_blood_pressure"] = ("High blood pressure (yes/no answer) by BMI category",
                f"SELECT {r} AS bmi_category, COUNT(*) AS patients, "
                f"ROUND(100.0*AVG(CASE WHEN UPPER({h})='YES' THEN 1.0 ELSE 0.0 END),2) AS high_bp_pct "
                f"FROM {VIEW} GROUP BY {r} ORDER BY patients DESC")
        else:
            q["7_blood_pressure"] = ("Blood pressure analysis", None)

        # one scan: a single row with a missing count per column (turned into a long table in run_all)
        cols = list(self.df.columns)
        exprs = ["COUNT(*) AS total_rows"] + [
            f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS m_{i}" for i, c in enumerate(cols)]
        q["8_missing_values"] = ("Missing-value analysis per column", f"SELECT {', '.join(exprs)} FROM {VIEW}")

        if self._c("risk_label"):
            r = self._c("risk_label")
            q["9_risk_category"] = ("Risk-category distribution",
                f"SELECT COALESCE({r}, 'Missing') AS risk_category, COUNT(*) AS patients, "
                f"ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM {VIEW}),2) AS pct_of_records "
                f"FROM {VIEW} GROUP BY COALESCE({r}, 'Missing') ORDER BY patients DESC")
        else:
            q["9_risk_category"] = ("Risk-category distribution", None)
        return q

    def run(self, name: str, sql: str) -> "pandas.DataFrame":
        t = time.time()
        out = self.spark.sql(sql).toPandas()
        log.info("SQL %s: %d rows in %.2f s", name, len(out), time.time() - t)
        return out

    def _missing_long(self, wide) -> "pandas.DataFrame":
        import pandas as pd
        if wide.empty or pd.isna(wide["total_rows"].iloc[0]):
            total = 0
        else:
            total = int(wide["total_rows"].iloc[0])
        rows = []
        for i, c in enumerate(self.df.columns):
            val = wide[f"m_{i}"].iloc[0] if f"m_{i}" in wide.columns else 0
            miss = int(val) if pd.notna(val) else 0
            rows.append({"column_name": c, "missing": miss, "total": total,
                         "missing_pct": round(100.0 * miss / total, 2) if total else 0.0})
        return pd.DataFrame(rows)

    def run_all(self) -> dict:
        """name -> {title, sql, frame, seconds, error}. A missing column is reported, never guessed."""
        results = {}
        for name, (title, sql) in self.queries().items():
            if sql is None:
                results[name] = {"title": title, "sql": None, "frame": None, "seconds": 0.0,
                                 "error": "columns for this query are not in the dataset"}
                continue
            t = time.time()
            frame = self.spark.sql(sql).toPandas()
            if name.startswith("8_"):
                frame = self._missing_long(frame)
            secs = round(time.time() - t, 3)
            log.info("SQL %s: %d rows in %.2f s", name, len(frame), secs)
            results[name] = {"title": title, "sql": sql, "frame": frame, "seconds": secs, "error": None}
        return results
