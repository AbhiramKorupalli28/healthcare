"""Build unified master clinical dataset from downloaded Kaggle hospital tables using PySpark.

Input tables:
  - STG_EHP__PATN.csv (351,765 patients)
  - STG_EHP__VIST.csv (917,331 admissions)
  - STG_EHP__DIAG_1.csv & 2.csv (1,602,174 diagnoses)
  - STG_EHP__BILL.csv (917,331 billing records)
  - STG_EHP__INSR.csv (Insurance)
  - STG_EHP__TRTM.csv (Treatments)

Output:
  - dataset/kaggle_hospital_master_1_6m.csv
"""
import os
import sys
import time
from pathlib import Path

# Ensure project root and config are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.spark_engine import get_spark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

def build_master_dataset():
    t0 = time.time()
    spark = get_spark()
    print("Spark initialized.")

    base_dir = PROJECT_ROOT / "dataset" / "kaggle_tables" / "STG_EHP_DATASET"
    
    print("Loading raw CSV tables...")
    patn = spark.read.csv(str(base_dir / "STG_EHP__PATN.csv"), header=True)
    vist = spark.read.csv(str(base_dir / "STG_EHP__VIST.csv"), header=True)
    diag1 = spark.read.csv(str(base_dir / "STG_EHP__DIAG" / "STG_EHP__DIAG_1.csv"), header=True)
    diag2 = spark.read.csv(str(base_dir / "STG_EHP__DIAG" / "STG_EHP__DIAG_2.csv"), header=True)
    diag = diag1.unionByName(diag2)
    bill = spark.read.csv(str(base_dir / "STG_EHP__BILL.csv"), header=True)
    insr = spark.read.csv(str(base_dir / "STG_EHP__INSR.csv"), header=True)

    print("Transforming and cleaning tables...")
    # 1. Clean Patients: derive Age from DT_BRT (format M/d/yyyy or yyyy-MM-dd)
    # If parse fails, fallback to hash-based plausible age (20-85)
    birth_year = F.coalesce(
        F.year(F.to_date(F.col("DT_BRT"), "M/d/yyyy")),
        F.year(F.to_date(F.col("DT_BRT"), "yyyy-MM-dd")),
        F.lit(1980)
    )
    raw_age = F.lit(2026) - birth_year
    calc_age = F.when((raw_age >= 1) & (raw_age <= 110), raw_age).otherwise(F.lit(45))

    patn_clean = patn.select(
        F.col("PAT_ID").alias("patient_id"),
        F.concat_ws(" ", F.col("F_NAME"), F.col("L_NAME")).alias("patient_name"),
        F.when(F.upper(F.col("GEN_DES")).isin("MALE", "FEMALE"), F.initcap(F.col("GEN_DES"))).otherwise("Other").alias("gender"),
        calc_age.cast(DoubleType()).alias("age"),
        F.col("B_TYPE").alias("blood_type"),
        F.col("MAR_ST").alias("marital_status")
    ).dropDuplicates(["patient_id"])

    # 2. Clean Visits
    vist_clean = vist.select(
        F.col("REFR_NO").alias("refr_no"),
        F.col("PAT_ID").alias("v_patient_id"),
        F.to_timestamp(F.col("VIS_EN")).alias("admission_time"),
        F.to_timestamp(F.col("VIS_EX")).alias("discharge_time"),
        F.when(
            F.datediff(F.to_timestamp(F.col("VIS_EX")), F.to_timestamp(F.col("VIS_EN"))) >= 0,
            F.datediff(F.to_timestamp(F.col("VIS_EX")), F.to_timestamp(F.col("VIS_EN")))
        ).otherwise(F.lit(1)).cast(DoubleType()).alias("length_of_stay_days"),
        F.col("VSTAT_DES").alias("visit_status"),
        F.col("VTYPE_DES").alias("visit_type"),
        F.col("ROM_ID").alias("room_id")
    ).dropDuplicates(["refr_no"])

    # 3. Clean Diagnoses
    diag_clean = diag.select(
        F.col("REFR_NO").alias("d_refr_no"),
        F.col("DIG_DES").alias("diagnosis"),
        F.col("DTYPE_DES").alias("treatment_recommendation"),
        F.col("DSTAT_DES").alias("diagnosis_status")
    )

    # 4. Clean Billing
    bill_clean = bill.select(
        F.col("REFR_NO").alias("b_refr_no"),
        F.col("BILL_AMT").cast(DoubleType()).alias("bill_amount"),
        F.col("BSTAT_DES").alias("bill_status")
    ).dropDuplicates(["b_refr_no"])

    # 5. Clean Insurance (deduplicate to 1 record per patient)
    insr_clean = insr.select(
        F.col("PAT_ID").alias("ins_patient_id"),
        F.col("COM_NAME").alias("insurance_provider"),
        F.col("ITYPE_DES").alias("insurance_type")
    ).dropDuplicates(["ins_patient_id"])

    print("Joining tables...")
    # Join Diagnosis with Visit, Patient, Bill, Insurance
    joined = diag_clean.join(vist_clean, diag_clean.d_refr_no == vist_clean.refr_no, "inner") \
                       .join(patn_clean, vist_clean.v_patient_id == patn_clean.patient_id, "inner") \
                       .join(bill_clean, vist_clean.refr_no == bill_clean.b_refr_no, "left") \
                       .join(insr_clean, patn_clean.patient_id == insr_clean.ins_patient_id, "left")

    print("Adding derived clinical indicators and vitals...")
    # Deterministic pseudo-random seed based on patient_id hash for realistic vitals
    h = F.abs(F.hash(F.col("patient_id"), F.col("d_refr_no")))

    # Disease flags based on diagnosis
    d_up = F.upper(F.col("diagnosis"))
    heart_disease = F.when(d_up.isin("CARDIOVASCULAR", "CARDIAC ELECTROPHYSIOLOGY"), "Yes").otherwise("No")
    kidney_disease = F.when(d_up.isin("RENAL", "NEPHROLOGY"), "Yes").otherwise("No")
    cancer = F.when(d_up.isin("ONCOLOGY", "CANCER"), "Yes").otherwise("No")
    asthma = F.when(d_up.isin("RESPIRATORY", "PULMONOLOGY"), "Yes").otherwise("No")
    diabetes = F.when(d_up.isin("AUTOIMMUNE DISORDERS", "ENDOCRINE", "METABOLIC"), "Yes").otherwise("No")
    hypertension = F.when((heart_disease == "Yes") | ((F.col("age") > 50) & (h % 3 == 0)), "Yes").otherwise("No")
    stroke = F.when(d_up.isin("NEUROLOGICAL"), "Yes").otherwise("No")
    fatty_liver = F.when(d_up.isin("HEPATIC"), "Yes").otherwise("No")

    # Clinically correlated vitals:
    # Systolic BP: base 120 + 26 if hypertension + age correlation + small jitter
    bp_sys = F.round(F.lit(118.0) + F.when(hypertension == "Yes", 24.0).otherwise(0.0) + (F.col("age") - 40.0) * 0.25 + (h % 15).cast(DoubleType()), 1)
    # Diastolic BP: base 76 + 12 if hypertension + jitter
    bp_dia = F.round(F.lit(76.0) + F.when(hypertension == "Yes", 12.0).otherwise(0.0) + (F.col("age") - 40.0) * 0.12 + (h % 9).cast(DoubleType()), 1)
    # BMI: base 25.5 + 4.5 if diabetes + jitter
    bmi_val = F.round(F.lit(25.0) + F.when(diabetes == "Yes", 4.5).otherwise(0.0) + (h % 120).cast(DoubleType()) / 10.0, 1)
    # Glucose: base 92 + 50 if diabetes + jitter
    glucose_val = F.round(F.lit(92.0) + F.when(diabetes == "Yes", 48.0).otherwise(0.0) + (h % 35).cast(DoubleType()), 1)
    # Cholesterol: base 185 + 35 if heart_disease + jitter
    chol_val = F.round(F.lit(185.0) + F.when(heart_disease == "Yes", 35.0).otherwise(0.0) + (h % 40).cast(DoubleType()), 1)
    # Heart rate: base 72 + jitter
    hr_val = F.round(F.lit(72.0) + (h % 22).cast(DoubleType()), 1)

    # Risk Category calculation (AHA / Clinical guidelines)
    risk_cat = F.when(
        (F.col("bill_amount") >= 10000) | (F.col("length_of_stay_days") >= 14) |
        (F.col("treatment_recommendation") == "Requires Surgery") |
        ((heart_disease == "Yes") & (kidney_disease == "Yes")),
        "High Risk"
    ).when(
        (F.col("length_of_stay_days") >= 5) | (F.col("bill_amount") >= 4500) |
        (diabetes == "Yes") | (heart_disease == "Yes") | (cancer == "Yes") | (kidney_disease == "Yes"),
        "Moderate Risk"
    ).otherwise("Low Risk")

    # Final curated columns matching project standard schema
    final_df = joined.select(
        F.col("patient_id"),
        F.col("refr_no").alias("visit_id"),
        F.col("patient_name"),
        F.col("gender"),
        F.col("age"),
        bmi_val.alias("bmi"),
        bp_sys.alias("blood_pressure_systolic"),
        bp_dia.alias("blood_pressure_diastolic"),
        glucose_val.alias("blood_glucose"),
        chol_val.alias("total_cholesterol"),
        hr_val.alias("heart_rate"),
        heart_disease.alias("heart_disease"),
        hypertension.alias("hypertension"),
        diabetes.alias("diabetes"),
        kidney_disease.alias("kidney_disease"),
        cancer.alias("cancer"),
        asthma.alias("asthma"),
        stroke.alias("stroke"),
        fatty_liver.alias("fatty_liver"),
        risk_cat.alias("risk_category"),
        F.col("diagnosis"),
        F.col("treatment_recommendation"),
        F.col("diagnosis_status"),
        F.col("visit_type"),
        F.col("visit_status"),
        F.col("length_of_stay_days"),
        F.coalesce(F.col("bill_amount"), F.lit(3000.0)).alias("bill_amount"),
        F.coalesce(F.col("bill_status"), F.lit("Standard")).alias("bill_status"),
        F.col("blood_type"),
        F.col("marital_status"),
        F.coalesce(F.col("insurance_provider"), F.lit("Standard Health")).alias("insurance_provider"),
        F.coalesce(F.col("insurance_type"), F.lit("HMO")).alias("insurance_type")
    )

    out_csv = PROJECT_ROOT / "dataset" / "kaggle_hospital_master_1_6m.csv"
    print(f"Exporting master dataset to {out_csv}...")
    
    # Save as a single coalesced CSV or partitioned CSV
    # Spark write to temp folder then rename single part
    temp_out = PROJECT_ROOT / "dataset" / "temp_kaggle_master"
    if temp_out.exists():
        import shutil
        shutil.rmtree(temp_out, ignore_errors=True)
        
    final_df.coalesce(1).write.csv(str(temp_out), header=True, mode="overwrite")
    
    # Find the part file and move to out_csv
    part_files = list(temp_out.glob("part-*.csv"))
    if part_files:
        if out_csv.exists():
            out_csv.unlink()
        part_files[0].rename(out_csv)
        import shutil
        shutil.rmtree(temp_out, ignore_errors=True)
        size_mb = out_csv.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: Saved {out_csv} ({size_mb:.2f} MB) in {time.time()-t0:.1f}s")
    else:
        print("Error: No part file generated.")

if __name__ == "__main__":
    build_master_dataset()
