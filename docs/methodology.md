# Project Methodology & Architecture

This document describes the methodology, distributed pipeline architecture, and implementation details for the Healthcare Big Data Analytics project.

---

## 1. Pipeline Overview

The project processes clinical and epidemiological datasets at scale using **Apache Spark (PySpark)** and **Hadoop HDFS**. The pipeline is structured into five sequential stages:

```
Raw CSV / Staging Tables
          │
          ▼
1. Distributed Ingestion & Schema Alignment (PySpark)
          │
          ▼
2. Data Cleaning, Validation & Parquet Conversion
          │
          ▼
3. Distributed Spark SQL Analytical Queries (9 Queries)
          │
          ▼
4. Scalable Machine Learning Classification & Clustering
          │
          ▼
5. Real-Time Interactive Web Dashboard (Streamlit)
```

---

## 2. Distributed Data Ingestion & Storage

- **Hadoop HDFS Ingestion:** Raw datasets are loaded into HDFS under `/healthcare/input`. When running in standalone mode or without an active cluster, an automatic local storage fallback ensures uninterrupted processing.
- **Relational Joining:** For multi-table hospital datasets (Kaggle 20-table schema), PySpark performs distributed joins between:
  - `STG_EHP__PATN` (Patients): Demographics, DOB, gender, blood type, marital status.
  - `STG_EHP__VIST` (Admissions): Entry and exit timestamps, admission type, room assignments.
  - `STG_EHP__DIAG` (Diagnoses): ICD condition codes, descriptions, and treatment categories.
  - `STG_EHP__BILL` (Billing): Billing amounts and payment statuses.
  - `STG_EHP__INSR` (Insurance): Policy coverage and provider details.

---

## 3. Data Cleaning & Validation

The PySpark cleaning pipeline applies deterministic transformations across all partitions:
1. **Column Standardisation:** All column names are normalized into uniform `snake_case`.
2. **Deduplication:** Exact duplicate rows are detected and removed.
3. **Type Casting & Null Handling:** Text fields are stripped of trailing whitespaces; empty strings are converted to `NULL`. Numeric metrics are cast to double precision.
4. **Validity Rules:** Records outside plausible clinical boundaries are filtered into a separate `rejected.parquet` partition:
   - Patient age: $0 \le \text{age} \le 120$
   - BMI: $10 \le \text{BMI} \le 100$
   - Blood pressure: Systolic $\le$ Diastolic checks
5. **Derived Metrics:** Calculates patient age from birth dates, lengths of stay in days ($\text{discharge} - \text{admission}$), and WHO BMI categories.

---

## 4. Analytical Spark SQL Queries

An in-memory temporary view (`healthcare_patients`) is queried using Spark SQL across nine dimensions:
1. **Total Records:** Cleaned and valid record volume.
2. **Disease Frequencies:** Positive and known counts across chronic conditions.
3. **Age Group Distribution:** Stratified prevalence across age buckets ($<30$, $30-39$, $40-49$, $50-59$, $60-69$, $70+$).
4. **Gender-wise Distribution:** Prevalence comparison across genders.
5. **BMI by Disease:** Mean BMI stratified by positive vs. negative condition flags.
6. **Blood Glucose by Disease:** Average glucose levels across chronic illnesses.
7. **Blood Pressure Stratification:** AHA 2017 categories (Normal, Elevated, Stage 1, Stage 2).
8. **Missing Values Analysis:** Column-by-column missingness profile.
9. **Risk Stratification:** Classification into Low, Moderate, and High clinical risk cohorts.

---

## 5. Machine Learning Methodology

- **Feature Selection & Leakage Prevention:** Outcome indicators and derived diagnosis labels are strictly isolated from predictor feature sets.
- **Stratified Train/Test Split:** 80% training, 20% testing with fixed random seed ($42$).
- **Handling Class Imbalance:** Models use balanced class weighting to ensure rare conditions are properly penalized during optimization.
- **Algorithms:**
  - **Random Forest Classifier:** 200 decision trees with feature importance calculation.
  - **Decision Tree Classifier:** Maximum depth 6 for interpretable baseline comparisons.
  - **K-Means Clustering:** Unsupervised patient cohort discovery on standardized continuous features ($k=3$).
