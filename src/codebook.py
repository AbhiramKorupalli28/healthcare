"""BRFSS codebook: readable names and code labels for the CDC survey variables.

Mapping follows the CDC BRFSS codebook. Codes outside each map (7 and 9 are
"don't know" / "refused", 77 and 99 are not asked / refused) become null. A
variable is only touched when it is present in the file, so other CSVs pass
through unchanged.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# CDC variable name -> readable name used throughout the project
RENAME = {
    "_AGE80": "age", "SEXVAR": "gender", "_BMI5": "bmi", "_BMI5CAT": "risk_category",
    "DIABETE4": "diabetes", "BPHIGH6": "hypertension", "TOLDHI3": "high_cholesterol",
    "CVDCRHD4": "heart_disease", "CVDINFR4": "heart_attack", "CVDSTRK3": "stroke",
    "ASTHMA3": "asthma", "CHCKIDNY": "kidney_disease", "CHCCOPD3": "copd", "CHCOCNCR": "cancer",
    "GENHLTH": "general_health", "PHYSHLTH": "physical_health_days", "MENTHLTH": "mental_health_days",
    "_TOTINDA": "physically_active", "SLEPTIM1": "sleep_hours", "_SMOKER3": "smoking_status",
    "_LLCPWT": "survey_weight", "_RFHYPE6": "hypertension_risk_flag", "_RFCHOL3": "cholesterol_risk_flag",
    "_RFBMI5": "overweight_risk_flag",
}

YES_NO = {1: "Yes", 2: "No"}
# code -> label, per variable (readable name)
CODE_MAPS = {
    "gender": {1: "Male", 2: "Female"},
    "diabetes": {1: "Yes", 2: "Yes", 3: "No", 4: "Borderline"},          # 2 = yes, only during pregnancy
    "hypertension": {1: "Yes", 2: "Yes", 3: "Borderline", 4: "No"},      # 2 = yes, only during pregnancy
    "risk_category": {1: "Underweight", 2: "Normal weight", 3: "Overweight", 4: "Obese"},
    "general_health": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "smoking_status": {1: "Current (every day)", 2: "Current (some days)", 3: "Former", 4: "Never"},
    "hypertension_risk_flag": {1: "No", 2: "Yes"},
    "cholesterol_risk_flag": {1: "No", 2: "Yes"},
    "overweight_risk_flag": {1: "No", 2: "Yes"},
}
for _name in ["high_cholesterol", "heart_disease", "heart_attack", "stroke", "asthma",
              "kidney_disease", "copd", "cancer", "physically_active"]:
    CODE_MAPS[_name] = YES_NO

DAYS_VARS = ["physical_health_days", "mental_health_days"]  # 1-30 days; 88 = none; 77/99 = DK/refused


def apply_codebook(df: DataFrame) -> tuple[DataFrame, dict]:
    """Rename, label and clean the CDC variables that are present. Returns (df, what_changed)."""
    changed = {"renamed": [], "labelled": [], "scaled": [], "special": []}
    cols = set(df.columns)

    # Only apply CDC BRFSS codebook transformations if raw CDC survey columns are present
    is_brfss = any(c in RENAME for c in cols)
    if not is_brfss:
        return df, changed

    for old, new in RENAME.items():
        if old in cols and new not in cols:
            df = df.withColumnRenamed(old, new)
            cols.discard(old)
            cols.add(new)
            changed["renamed"].append(f"{old}->{new}")
    for name, mapping in CODE_MAPS.items():
        if name in cols:
            pairs = [x for k, v in mapping.items() for x in (F.lit(k), F.lit(v))]
            df = df.withColumn(name, F.create_map(*pairs)[F.col(name).cast("int")])
            changed["labelled"].append(name)
    if "bmi" in cols and ("_BMI5" in changed["renamed"]):
        # BRFSS stores BMI with two implied decimals (2700 = 27.00)
        df = df.withColumn("bmi", F.col("bmi").cast("double") / 100.0)
        changed["scaled"].append("bmi / 100")
    for name in DAYS_VARS:
        if name in cols:
            c = F.col(name).cast("int")
            df = df.withColumn(name, F.when(c == 88, F.lit(0)).when(c.isin(77, 99), None)
                               .otherwise(c.cast("double")))
            changed["special"].append(f"{name}: 88->0, 77/99->null")
    if "sleep_hours" in cols:
        c = F.col("sleep_hours").cast("int")
        df = df.withColumn("sleep_hours", F.when(c.isin(77, 99), None).otherwise(c.cast("double")))
        changed["special"].append("sleep_hours: 77/99->null")
    return df, changed
