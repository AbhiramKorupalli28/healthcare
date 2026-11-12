"""Central configuration for the Healthcare Analytics project.

Machine-specific values (HDFS address, Spark master, memory) are read from
environment variables with safe local defaults, so the same code runs on a
Windows laptop and on Linux.
"""
import os
import sys
from pathlib import Path

# Ensure PySpark workers match the current Python interpreter
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Safe local defaults for Java and Hadoop on Windows if not already in environment
if "JAVA_HOME" not in os.environ:
    for candidate in [Path.home() / ".jdk17" / "jdk-17.0.21+9", Path("C:/Program Files/Java/jdk-17")]:
        if candidate.exists():
            os.environ["JAVA_HOME"] = str(candidate)
            os.environ["PATH"] = f"{candidate / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            break

if "HADOOP_HOME" not in os.environ:
    candidate = Path("C:/hadoop")
    if candidate.exists():
        os.environ["HADOOP_HOME"] = str(candidate)
        os.environ["PATH"] = f"{candidate / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
UPLOAD_DIR = DATASET_DIR / "uploads"                  # CSVs uploaded in the dashboard
LOCAL_HDFS_DIR = DATASET_DIR / "hdfs_local_fallback"  # stand-in used ONLY when HDFS is unreachable
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEAN_DIR = OUTPUT_DIR / "cleaned"                    # cleaned Parquet + run_stats.json per dataset
SUMMARY_DIR = OUTPUT_DIR / "summaries"                # Spark SQL result CSVs
LOG_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"                # visual report PNGs

# The dataset of record: CDC BRFSS annual files 2021-2023 (built by scripts/build_dataset.py)
DEFAULT_DATASET = DATASET_DIR / "brfss_2021_2023_combined.csv"
DATASET_INFO = {
    "name": "CDC BRFSS 2021-2023 combined (shared variables)",
    "source_url": "https://www.cdc.gov/brfss/annual_data/annual_2022.html",
    "author": "Centers for Disease Control and Prevention (CDC)",
    "licence": "US federal government work (public data)",
    "synthetic": False,
    "publisher_note": "",
}

# HDFS (WebHDFS REST for file operations, hdfs:// URI for Spark I/O)
HDFS_NAMENODE_URL = os.environ.get("HDFS_NAMENODE_URL", "http://localhost:9870")
HDFS_FS_URI = os.environ.get("HDFS_FS_URI", "hdfs://localhost:9000")
HDFS_USER = os.environ.get("HDFS_USER", "user")
HDFS_TIMEOUT_S = 4
HDFS_INPUT_DIR = "/healthcare/input"
HDFS_PROCESSED_DIR = "/healthcare/processed"

# Spark
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "3g")
SPARK_SHUFFLE_PARTITIONS = 8

# Machine learning
RANDOM_STATE = 42
TEST_SIZE = 0.2
ML_MAX_ROWS = 50000          # rows sampled by Spark and passed to scikit-learn
RF_TREES = 200

# Dynamic column detection: each logical role maps to the header names it may carry.
# Names are compared after standardisation (lower case, non-alphanumerics -> "_").
COLUMN_ALIASES = {
    "patient_id": ["patient_id", "patientid", "id"],
    "age": ["age", "age80", "patient_age"],
    "gender": ["gender", "sex"],
    "bmi": ["bmi", "body_mass_index"],
    "height_cm": ["height_cm", "height"],
    "weight_kg": ["weight_kg", "weight"],
    "glucose": ["blood_glucose", "glucose", "glucose_level"],
    "hba1c": ["hba1c", "hba1c_level", "glycated_hemoglobin"],
    "bp_systolic": ["blood_pressure_systolic", "systolic_bp", "bp_systolic", "systolic"],
    "bp_diastolic": ["blood_pressure_diastolic", "diastolic_bp", "bp_diastolic", "diastolic"],
    "cholesterol": ["total_cholesterol", "cholesterol"],
    "hdl": ["hdl", "hdl_cholesterol"],
    "ldl": ["ldl", "ldl_cholesterol"],
    "triglycerides": ["triglycerides"],
    "heart_rate": ["heart_rate", "pulse"],
    "sleep_hours": ["sleep_hours"],
    "physical_health_days": ["physical_health_days"],
    "mental_health_days": ["mental_health_days"],
    "general_health": ["general_health"],
    "physically_active": ["physically_active"],
    "smoking_status": ["smoking_status"],
    "survey_weight": ["survey_weight"],
    "risk_label": ["risk_category", "diabetes_risk", "risk_label", "risk_level"],
    "risk_score": ["diabetes_risk_score", "risk_score"],
    "diabetes": ["diabetes"],
    "hypertension": ["hypertension"],
    "high_cholesterol": ["high_cholesterol"],
    "heart_disease": ["heart_disease"],
    "heart_attack": ["heart_attack"],
    "stroke": ["stroke"],
    "asthma": ["asthma"],
    "kidney_disease": ["kidney_disease"],
    "copd": ["copd"],
    "cancer": ["cancer"],
    "fatty_liver": ["fatty_liver"],
    "pcos": ["pcos"],
    "hypertension_risk_flag": ["hypertension_risk_flag"],
    "cholesterol_risk_flag": ["cholesterol_risk_flag"],
    "overweight_risk_flag": ["overweight_risk_flag"],
}
# Yes/No condition flags, used for disease frequency and disease-wise comparisons.
DISEASE_ROLES = ["diabetes", "hypertension", "high_cholesterol", "heart_disease", "heart_attack",
                 "stroke", "asthma", "kidney_disease", "copd", "cancer"]
# Targets offered to the models, in order, when the role is present in the file
ML_TARGET_ROLES = ["risk_label", "diabetes", "heart_disease", "hypertension", "high_cholesterol"]
# Measures and attributes used as model inputs when present (curated, documented in the README).
FEATURE_ROLES = ["age", "gender", "bmi", "general_health", "physical_health_days", "mental_health_days",
                 "physically_active", "sleep_hours", "smoking_status"] + DISEASE_ROLES
# Columns that restate the outcome and would leak it: never features for any target.
LEAKAGE_ROLES = ["risk_score", "risk_label"]
# Per target column (standardised name): columns derived from that target or from its own measure.
TARGET_LEAKAGE = {
    "risk_category": ["bmi", "overweight_risk_flag"],   # BMI category is computed from BMI
    "hypertension": ["hypertension_risk_flag"],          # CDC risk flag is computed from the same answer
    "high_cholesterol": ["cholesterol_risk_flag"],
}
NON_FEATURE_ROLES = ["patient_id", "survey_weight"]
LEAKAGE_NAME_HINTS = ["recommendation", "consultation", "risk_score", "risk_category", "risk_label"]

# Plausible physiological / survey ranges; a value outside its range marks the record invalid.
VALID_RANGES = {
    "age": (0, 120), "bmi": (10, 100), "glucose": (20, 600), "hba1c": (3, 20),
    "bp_systolic": (50, 260), "bp_diastolic": (30, 200), "cholesterol": (50, 600),
    "heart_rate": (20, 250), "physical_health_days": (0, 30), "mental_health_days": (0, 30),
    "sleep_hours": (1, 24),
}

# Theme (Modern Light SaaS with Forest Green accent, matching user specification)
THEME = {
    "bg": "#F8FAFC", "panel": "#FFFFFF", "panel2": "#F1F5F9", "text": "#111827",
    "muted": "#6B7280", "forest_green": "#0E4D2A", "green": "#10B981", "light_green": "#EBF5EE",
    "border": "#E5E7EB", "blue": "#2563EB", "amber": "#D97706", "red": "#DC2626",
    "violet": "#0E4D2A", "lilac": "#059669",
}
