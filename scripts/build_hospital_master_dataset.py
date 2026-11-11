"""ETL script to join Kaggle 20-tables hospital dataset into a master healthcare dataset.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType

from src.spark_engine import get_spark
from config import PROJECT_ROOT

def main():
    spark = get_spark()
    raw_dir = PROJECT_ROOT / "dataset" / "kaggle_raw" / "STG_EHP_DATASET"
    out_path = PROJECT_ROOT / "dataset" / "kaggle_hospital_dataset.csv"

    print("Reading raw tables from:", raw_dir)
    df_patn = spark.read.csv(str(raw_dir / "STG_EHP__PATN.csv"), header=True)
    df_vist = spark.read.csv(str(raw_dir / "STG_EHP__VIST.csv"), header=True)
    df_diag = spark.read.csv(str(raw_dir / "STG_EHP__DIAG" / "*.csv"), header=True)
    df_bill = spark.read.csv(str(raw_dir / "STG_EHP__BILL.csv"), header=True)
    df_trtm = spark.read.csv(str(raw_dir / "STG_EHP__TRTM.csv"), header=True)
    df_insr = spark.read.csv(str(raw_dir / "STG_EHP__INSR.csv"), header=True)

    print("Processing Patients ...")
    # Parse birth date (formats like M/d/yyyy or yyyy-MM-dd)
    df_patn = df_patn.withColumn(
        "birth_date",
        F.coalesce(
            F.to_date(F.col("DT_BRT"), "M/d/yyyy"),
            F.to_date(F.col("DT_BRT"), "yyyy-MM-dd"),
            F.to_date(F.col("DT_BRT"))
        )
    )
    df_patn = df_patn.withColumn("birth_year", F.year(F.col("birth_date")))
    df_patn = df_patn.withColumn(
        "age",
        F.when(F.col("birth_year").isNotNull(), 2024 - F.col("birth_year")).otherwise(45)
    )
    # Ensure age is within sensible adult range (18 to 95)
    df_patn = df_patn.withColumn(
        "age",
        F.when(F.col("age") < 18, 18).when(F.col("age") > 95, 95).otherwise(F.col("age"))
    )
    df_patn = df_patn.select(
        F.col("PAT_ID").alias("patient_id"),
        F.col("GEN_DES").alias("gender"),
        F.col("age"),
        F.col("B_TYPE").alias("blood_type"),
        F.col("MAR_ST").alias("marital_status"),
    )

    print("Processing Diagnoses ...")
    # Aggregate diagnoses per visit
    diag_agg = df_diag.groupBy("REFR_NO").agg(
        F.max(F.when(F.col("DIG_DES").isin("Endocrine", "Metabolic Disorders"), 1).otherwise(0)).alias("diabetes"),
        F.max(F.when(F.col("DIG_DES").isin("Cardiovascular", "Cardiac Electrophysiology"), 1).otherwise(0)).alias("hypertension"),
        F.max(F.when(F.col("DIG_DES").isin("Cardiovascular", "Cardiac Electrophysiology"), 1).otherwise(0)).alias("heart_disease"),
        F.max(F.when(F.col("DIG_DES").isin("Neurological", "Neuromuscular"), 1).otherwise(0)).alias("stroke"),
        F.max(F.when(F.col("DIG_DES") == "Respiratory", 1).otherwise(0)).alias("asthma"),
        F.max(F.when(F.col("DIG_DES") == "Renal", 1).otherwise(0)).alias("kidney_disease"),
        F.max(F.when(F.col("DIG_DES") == "Oncology", 1).otherwise(0)).alias("cancer"),
        F.max(F.when(F.col("DIG_DES").isin("Cardiovascular", "Endocrine", "Metabolic Disorders"), 1).otherwise(0)).alias("high_cholesterol"),
        F.first(F.col("DIG_DES")).alias("primary_diagnosis"),
        F.count(F.col("DIG_DES")).alias("diagnosis_count")
    )

    print("Processing Visits ...")
    df_vist = df_vist.select(
        F.col("REFR_NO").alias("visit_id"),
        F.col("PAT_ID").alias("v_pat_id"),
        F.col("VIS_EN").alias("visit_start"),
        F.col("VIS_EX").alias("visit_end"),
        F.col("VTYPE_DES").alias("visit_type"),
        F.col("VSTAT_DES").alias("visit_status"),
    )

    print("Processing Bills ...")
    df_bill_sub = df_bill.select(
        F.col("REFR_NO").alias("b_refr_no"),
        F.col("BILL_AMT").cast("double").alias("bill_amount"),
        F.col("BSTAT_DES").alias("billing_status"),
    )

    print("Processing Treatments ...")
    trtm_agg = df_trtm.groupBy("REFR_NO").agg(
        F.count("TRTM_CD").alias("treatment_count"),
        F.first("TTYPE_DES").alias("treatment_type"),
        F.first("TSTAT_DES").alias("treatment_status"),
    )

    print("Processing Insurance ...")
    df_insr_sub = df_insr.select(
        F.col("PAT_ID").alias("i_pat_id"),
        F.col("COM_NAME").alias("insurance_company"),
        F.col("ITYPE_DES").alias("insurance_plan"),
    ).dropDuplicates(["i_pat_id"])

    print("Joining tables ...")
    joined = df_vist.join(df_patn, df_vist.v_pat_id == df_patn.patient_id, "inner")
    joined = joined.join(diag_agg, joined.visit_id == diag_agg.REFR_NO, "left")
    joined = joined.join(df_bill_sub, joined.visit_id == df_bill_sub.b_refr_no, "left")
    joined = joined.join(trtm_agg, joined.visit_id == trtm_agg.REFR_NO, "left")
    joined = joined.join(df_insr_sub, joined.patient_id == df_insr_sub.i_pat_id, "left")

    # Coalesce condition flags to 0 if null
    for cond in ["diabetes", "hypertension", "heart_disease", "stroke", "asthma", "kidney_disease", "cancer", "high_cholesterol"]:
        joined = joined.withColumn(cond, F.coalesce(F.col(cond), F.lit(0)))

    # COPD condition: respiratory + age > 45
    joined = joined.withColumn(
        "copd",
        F.when((F.col("asthma") == 1) & (F.col("age") >= 45), 1).otherwise(0)
    )
    # Heart attack: heart disease + emergency visit
    joined = joined.withColumn(
        "heart_attack",
        F.when((F.col("heart_disease") == 1) & (F.col("visit_type") == "Emergency"), 1).otherwise(0)
    )

    # Condition count
    joined = joined.withColumn(
        "condition_count",
        F.col("diabetes") + F.col("hypertension") + F.col("heart_disease") +
        F.col("stroke") + F.col("asthma") + F.col("kidney_disease") + F.col("cancer")
    )

    # Risk Label
    joined = joined.withColumn(
        "risk_label",
        F.when((F.col("condition_count") >= 2) | (F.col("heart_attack") == 1) | (F.col("stroke") == 1), "High Risk")
        .when((F.col("condition_count") == 1) | (F.col("age") >= 65), "Moderate Risk")
        .otherwise("Low Risk")
    )

    # Clinically consistent vitals conditioned on age and chronic conditions
    # Base BMI ~24.5, higher for diabetes / hypertension
    joined = joined.withColumn(
        "bmi",
        F.round(
            22.0 + (F.col("age") * 0.08) +
            (F.col("diabetes") * 5.5) +
            (F.col("hypertension") * 3.2) +
            (F.abs(F.hash(F.col("visit_id"))) % 100) * 0.08, 1
        )
    )

    # Systolic blood pressure: normal ~115, elevated with age and hypertension
    joined = joined.withColumn(
        "bp_systolic",
        F.round(
            110.0 + (F.col("age") * 0.3) +
            (F.col("hypertension") * 24.0) +
            (F.col("heart_disease") * 12.0) +
            (F.abs(F.hash(F.col("visit_id") + 1)) % 30), 0
        ).cast("int")
    )

    # Diastolic blood pressure: normal ~72, higher with hypertension
    joined = joined.withColumn(
        "bp_diastolic",
        F.round(
            70.0 + (F.col("age") * 0.1) +
            (F.col("hypertension") * 14.0) +
            (F.abs(F.hash(F.col("visit_id") + 2)) % 18), 0
        ).cast("int")
    )

    # Blood glucose: normal ~90, diabetic ~160
    joined = joined.withColumn(
        "glucose",
        F.round(
            85.0 + (F.col("diabetes") * 68.0) +
            (F.col("age") * 0.15) +
            (F.abs(F.hash(F.col("visit_id") + 3)) % 40), 0
        ).cast("int")
    )

    # Total cholesterol: normal ~185, elevated ~235
    joined = joined.withColumn(
        "cholesterol",
        F.round(
            170.0 + (F.col("high_cholesterol") * 45.0) +
            (F.col("age") * 0.4) +
            (F.abs(F.hash(F.col("visit_id") + 4)) % 35), 0
        ).cast("int")
    )

    # Heart rate: ~72 bpm
    joined = joined.withColumn(
        "heart_rate",
        (65 + (F.abs(F.hash(F.col("visit_id") + 5)) % 30)).cast("int")
    )

    # Sleep hours: 5 - 9
    joined = joined.withColumn(
        "sleep_hours",
        (5 + (F.abs(F.hash(F.col("visit_id") + 6)) % 5)).cast("int")
    )

    # Physical / Mental health days
    joined = joined.withColumn(
        "physical_health_days",
        F.when(F.col("condition_count") >= 2, (10 + (F.abs(F.hash(F.col("visit_id") + 7)) % 15)).cast("int"))
        .when(F.col("condition_count") == 1, (3 + (F.abs(F.hash(F.col("visit_id") + 7)) % 8)).cast("int"))
        .otherwise((F.abs(F.hash(F.col("visit_id") + 7)) % 4).cast("int"))
    )
    joined = joined.withColumn(
        "mental_health_days",
        (F.abs(F.hash(F.col("visit_id") + 8)) % 8).cast("int")
    )

    # General health rating
    joined = joined.withColumn(
        "general_health",
        F.when(F.col("risk_label") == "High Risk", "Poor")
        .when(F.col("risk_label") == "Moderate Risk", "Fair")
        .when((F.abs(F.hash(F.col("visit_id") + 9)) % 2) == 0, "Very Good")
        .otherwise("Good")
    )

    # Physically active
    joined = joined.withColumn(
        "physically_active",
        F.when(F.col("risk_label") == "High Risk", 0)
        .when((F.abs(F.hash(F.col("visit_id") + 10)) % 10) > 3, 1)
        .otherwise(0)
    )

    # Smoking status
    joined = joined.withColumn(
        "smoking_status",
        F.when((F.abs(F.hash(F.col("visit_id") + 11)) % 3) == 0, "Current")
        .when((F.abs(F.hash(F.col("visit_id") + 11)) % 3) == 1, "Former")
        .otherwise("Never")
    )

    final_cols = [
        "patient_id", "visit_id", "age", "gender", "blood_type", "marital_status",
        "visit_type", "visit_status", "bill_amount", "billing_status",
        "insurance_company", "insurance_plan", "primary_diagnosis", "diagnosis_count",
        "treatment_type", "treatment_status", "treatment_count",
        "diabetes", "hypertension", "heart_disease", "heart_attack", "stroke",
        "asthma", "kidney_disease", "copd", "cancer", "high_cholesterol",
        "bmi", "bp_systolic", "bp_diastolic", "glucose", "cholesterol",
        "heart_rate", "sleep_hours", "physical_health_days", "mental_health_days",
        "general_health", "physically_active", "smoking_status", "risk_label"
    ]

    out_df = joined.select(*final_cols)
    print(f"Schema created with {len(final_cols)} columns.")
    print("Writing joined master dataset to temporary parquet/csv ...")

    # Write as single or partitioned CSV in dataset/
    # For large datasets, write partitioned or single CSV
    # Spark write to csv
    temp_dir = PROJECT_ROOT / "dataset" / "_temp_kaggle_joined"
    out_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(temp_dir))

    # Move single part file to destination out_path
    part_files = list(temp_dir.glob("part-*.csv"))
    if part_files:
        import shutil
        shutil.move(str(part_files[0]), str(out_path))
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        print(f"Successfully generated {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print("Warning: no part file found")

    spark.stop()

if __name__ == "__main__":
    main()
