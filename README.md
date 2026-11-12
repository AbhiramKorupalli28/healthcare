# Healthcare Analytics for Disease Pattern Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.3-E25A1C.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io/)
[![Hadoop](https://img.shields.io/badge/Hadoop%20HDFS-3.3.6-66CCFF.svg)](https://hadoop.apache.org/)
[![GitHub](https://img.shields.io/badge/GitHub-healthcare-181717.svg)](https://github.com/AbhiramKorupalli28/healthcare.git)

A Big Data analytics project built for a B.Tech Big Data Analytics mini case study (roll A23126552281). It processes large-scale clinical and epidemiological healthcare data with **Apache Spark (PySpark, Spark SQL)**, stores data in **Hadoop HDFS**, trains **scikit-learn** models, and presents the results in an interactive **Streamlit** dashboard.

Repository: **[https://github.com/AbhiramKorupalli28/healthcare.git](https://github.com/AbhiramKorupalli28/healthcare.git)**

Every number in this README and in the dashboard comes from actual runs on the datasets. Run outputs are recorded in `output/run_results_local.json` and `output/run_results_hdfs.json`.

---

## 1. Problem Statement

Modern health records describe vast patient populations across multi-table clinical hospital systems and public surveys. However, actionable patterns—such as multimorbidity, condition prevalence across age groups, length of hospital stays, and high-risk patient trajectories—remain buried in datasets too large for standard spreadsheets or single-threaded tools. 

This project demonstrates how to store, clean, join, query, and model large-scale healthcare data (1M+ to 1.6M+ records, 1–2+ GB in-memory distributed processing) reliably using distributed computing (PySpark, Spark SQL, Hadoop HDFS) and serve insights through an intuitive web interface.

## 2. Objectives

1. **Distributed Storage:** Ingest and manage multi-hundred-megabyte to multi-gigabyte healthcare files in distributed storage (Hadoop HDFS with WebHDFS REST API and local fallback).
2. **Distributed Data Pipeline:** Clean, deduplicate, validate, and convert raw CSV data into partitioned columnar Parquet with PySpark, tracking full before/after reconciliation stats.
3. **Clinical & Epidemiological SQL Analytics:** Execute 9 Spark SQL queries analyzing disease counts, age/gender distributions, clinical vitals (BMI, blood pressure, glucose), and missing value profiles.
4. **Machine Learning at Scale:** Train and evaluate classification models (Decision Trees, Random Forests) and unsupervised K-Means clustering with leakage prevention and balanced class weights.
5. **Interactive Dashboard & Multi-Dataset Support:** Present live metrics, interactive Plotly visualizations, dataset upload capabilities (up to 4 GB), and downloadable reports via Streamlit.

## 3. Technologies

| Area | Technology | Version / Details |
|---|---|---|
| **Language** | Python | 3.11.x (tested on Windows 11 & Linux) |
| **Java Runtime** | OpenJDK | Eclipse Temurin 17.0.21+9 |
| **Distributed Processing** | Apache Spark | 3.5.3 via PySpark (local mode `local[*]`, all CPU cores) |
| **Distributed Storage** | Hadoop HDFS | 3.3.6 (single-node pseudo-distributed cluster + winutils) |
| **Analytical Engine** | Spark SQL | Temporary in-memory views with ANSI SQL compliance |
| **Machine Learning** | scikit-learn | 1.9.1 (DecisionTreeClassifier, RandomForestClassifier, KMeans) |
| **Visualizations** | Plotly & Matplotlib | Interactive dark-themed charts and visual report exports |
| **Web Dashboard** | Streamlit | 1.67.0 with custom multi-page routing and CSS styling |
| **Storage Formats** | CSV & Parquet | Input CSVs, partitioned Snappy-compressed Parquet |
| **Testing** | pytest | 9.2.0 (23 unit and integration tests) |

---

## 4. System Architecture

```
                      DATA SOURCES
   ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
   │ CDC BRFSS Multi-Year Survey Files │    │  Kaggle 20-Table Hospital System │
   │ (2021-2023, 1.31M records, 821 MB)│    │ (Patients, Visits, Diag, Billing)│
   └─────────────────┬─────────────────┘    └─────────────────┬────────────────┘
                     │                                        │
           scripts/build_dataset.py             scripts/build_kaggle_master_dataset.py
                     │                                        │
                     ▼                                        ▼
   dataset/brfss_2021_2023_combined.csv         dataset/kaggle_hospital_master_1_6m.csv
                                                             (1.602M rows, 32 cols)
                                         │
                   upload to HDFS  (src/hdfs_manager.py, WebHDFS REST)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            HDFS /healthcare/input/                 Local Storage Fallback
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                                PYSPARK PIPELINE
                ┌─────────────────────────────────────────────┐
                │ 1. Schema inference & snake_case conversion │
                │ 2. Deduplication & data profiling           │
                │ 3. Validity range enforcement               │
                │ 4. Derived clinical metrics & age groupings │
                └──────────────────────┬──────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
     cleaned.parquet (Valid)                               rejected.parquet (Invalid)
            │
    ┌───────┴──────────────────────────┬──────────────────────────┬──────────────────────┐
    ▼                                  ▼                          ▼                      ▼
Spark SQL (9 Queries)         Spark Aggregations         ML Engine (50K Sample)    Dashboard
src/sql_analytics.py          src/analytics.py           scikit-learn Random Forest  Streamlit App
output/summaries/*.csv        Overview & Disease Views   Decision Tree & K-Means   http://localhost:8501
```

---

## 5. Datasets Supported

The project includes built-in support for two massive healthcare datasets:

### A. Kaggle Enterprise Hospital Multi-Table System (Recommended)
- **Source:** [Kaggle Healthcare Dataset (20 Tables)](https://www.kaggle.com/datasets/moid1234/health-care-data-set-20-tables)
- **Integration Script:** [`scripts/build_kaggle_master_dataset.py`](scripts/build_kaggle_master_dataset.py)
- **Tables Integrated:**
  - `STG_EHP__PATN.csv`: 351,765 Patients (Demographics, Age, Gender, Blood Type, Marital Status)
  - `STG_EHP__VIST.csv`: 917,331 Admissions (Admission/Discharge, Length of Stay, Visit Type, Room)
  - `STG_EHP__DIAG_1.csv` & `2.csv`: 1,602,174 Clinical Diagnoses & ICD descriptions
  - `STG_EHP__BILL.csv`: 917,331 Billing records and payment status
  - `STG_EHP__INSR.csv`: Insurance providers and policy coverage
- **Master Output File:** `dataset/kaggle_hospital_master_1_6m.csv`
  - **1,602,174 rows × 32 columns** (~405 MB on disk, **>1.5 GB in-memory Spark execution**).

### B. CDC BRFSS Population Survey
- **Source:** US Centers for Disease Control and Prevention (CDC BRFSS annual files 2021, 2022, 2023).
- **Integration Script:** [`scripts/build_dataset.py`](scripts/build_dataset.py)
- **Master Output File:** `dataset/brfss_2021_2023_combined.csv`
  - **1,317,148 rows × 194 columns** (821.08 MB).

---

## 6. Installation & Quickstart

### Prerequisites
- **Python 3.11**
- **Java 17 (OpenJDK / Eclipse Temurin)**
- Windows only: Hadoop 3.3.6 `winutils.exe` and `hadoop.dll`

### Step 1: Clone Repository
```bash
git clone https://github.com/AbhiramKorupalli28/healthcare.git
cd healthcare
```

### Step 2: Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run PySpark Big Data Pipeline
Run the distributed pipeline from the command line:

```bash
# Analyze the 1.6M Kaggle Hospital master dataset
python run_analysis.py --dataset dataset/kaggle_hospital_master_1_6m.csv --force

# Or run with HDFS distributed storage enabled
python run_analysis.py --dataset dataset/kaggle_hospital_master_1_6m.csv --hdfs
```

### Step 4: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 7. Machine Learning Performance

Machine learning models are trained on representative stratified samples drawn directly from the cleaned Spark partitions. All pipelines use strict feature leakage control (e.g. outcome flags and derived categories are excluded from predictors) and stratified cross-validation.

### Kaggle Hospital Dataset (1.6M Records):
| Target Condition | Majority Baseline | Decision Tree Accuracy | Random Forest Accuracy | Random Forest Macro F1 |
|---|---|---|---|---|
| **Heart Disease** | 94.98% | 89.98% | **96.92%** | **0.8444** |
| **Diabetes** | 94.96% | 49.53% | **92.28%** | **0.6620** |
| **Hypertension** | 83.18% | 77.45% | **89.72%** | **0.8290** |
| **Risk Category** | 98.60% | 10.46% | **51.88%** | 0.2396 |

### CDC BRFSS Dataset (1.31M Records):
| Target Condition | Majority Baseline | Decision Tree Accuracy | Random Forest Accuracy | Random Forest Macro F1 |
|---|---|---|---|---|
| **Heart Disease** | 94.50% | 78.50% | **93.35%** | **0.6892** |
| **Diabetes** | 85.39% | 70.30% | **79.39%** | **0.6161** |
| **Risk Category** | 35.37% | 37.46% | **39.28%** | **0.3159** |

---

## 8. Spark SQL Analytics (9 Queries)

All 9 queries run in parallel over the in-memory temporary view `healthcare_patients`:

1. `1_total_records.csv`: Total valid record counts after distributed deduplication.
2. `2_disease_counts.csv`: Condition-wise positive vs. total patient frequency.
3. `3_age_group_distribution.csv`: Disease prevalence across age groups (18–29, 30–39, 40–49, 50–59, 60–69, 70+).
4. `4_gender_distribution.csv`: Prevalence breakdown across Male, Female, and Other.
5. `5_avg_bmi_by_disease.csv`: Average BMI stratified by disease status.
6. `6_avg_glucose_by_disease.csv`: Average blood glucose levels stratified by condition.
7. `7_blood_pressure.csv`: Blood pressure categorization according to AHA 2017 clinical standards (Normal, Elevated, Stage 1 HTN, Stage 2 HTN).
8. `8_missing_values.csv`: Columnar missingness audit and null percentages across all attributes.
9. `9_risk_category.csv`: Population distribution across High, Moderate, and Low risk cohorts.

All query results are automatically exported to [`output/summaries/`](output/summaries/).

---

## 9. Streamlit Dashboard Features

- **Dynamic Dataset Switcher:** Instantly toggle between `kaggle_hospital_master_1_6m.csv`, `brfss_2021_2023_combined.csv`, or uploaded custom CSVs directly from the sidebar.
- **Large Dataset Uploads:** Supports drag-and-drop file uploads up to **4 GB** (`maxUploadSize = 4096` in `.streamlit/config.toml`).
- **Real-Time Visualizations:** High-performance Plotly charts rendered in an accessible dark medical theme.
- **Risk Prediction Form:** Interactive form to test individual risk scores and multi-disease probabilities on trained Random Forest classifiers.
- **HDFS Manager:** WebHDFS interface to inspect cluster health, browse `/healthcare/input` and `/healthcare/processed`, and trigger distributed file moves.
- **Report Generator:** One-click download of all Spark SQL CSVs and summary reports.

---

## 10. Automated Tests

The test suite validates data ingestion, cleaning logic, Spark SQL aggregations, and ML model outputs against real records:

```bash
pytest -q tests
```
**Results:** `22 passed, 1 skipped (HDFS cluster test)` in ~45 seconds.

---

## 11. Project Directory Structure

```
healthcare/
├── .streamlit/
│   └── config.toml                  # Streamlit port, dark theme & 4GB upload settings
├── pages/
│   ├── overview.py                  # Population overview & KPI metric cards
│   ├── dataset_explorer.py          # 100-row preview, Spark schema & null profiling
│   ├── disease_analytics.py         # Multi-condition filters & prevalence charts
│   ├── risk_analysis.py             # ML model evaluations & interactive risk predictor
│   ├── bigdata_processing.py        # HDFS & PySpark job telemetry & logs
│   └── reports.py                   # One-click report and CSV downloads
├── scripts/
│   ├── build_dataset.py             # CDC BRFSS multi-year ETL compiler
│   ├── build_kaggle_master_dataset.py# Distributed PySpark join for Kaggle hospital tables
│   └── download_kaggle_clinical.py  # Multi-threaded Kaggle API dataset downloader
├── src/
│   ├── analytics.py                 # Spark aggregation helper routines
│   ├── app_state.py                 # Session state & cached Spark sessions
│   ├── codebook.py                  # Epidemiological dictionary & CDC decoders
│   ├── data_loader.py               # Dynamic schema & role discovery
│   ├── hdfs_manager.py              # WebHDFS client with local storage fallback
│   ├── logger.py                    # Structured logging to output/logs/
│   ├── ml_engine.py                 # Scikit-learn Random Forest, Decision Tree & K-Means
│   ├── pipeline.py                  # End-to-end distributed cleaning pipeline
│   ├── preprocessing.py             # Deduplication, type casting & range validation
│   ├── spark_engine.py              # PySpark session management & Parquet reader
│   ├── sql_analytics.py             # The 9 Spark SQL queries
│   ├── ui.py                        # CSS styling & responsive metric components
│   └── visualization.py             # Dark-themed Plotly charts
├── tests/
│   ├── fixtures/sample_2000_rows.csv# Test fixture from real records
│   ├── test_data_loader.py
│   ├── test_hdfs_manager.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   └── test_spark_sql_ml.py
├── app.py                           # Main Streamlit application entry point
├── config.py                        # Project configurations, column aliases & thresholds
├── requirements.txt                 # Pinned dependencies
├── run_analysis.py                  # Command-line analytics execution script
├── .gitignore                       # Clean Git configuration
└── README.md                        # Documentation
```

---

## 12. Author & License

- **Author:** Abhiram Korupalli
- **Course / Mini Project:** B.Tech Big Data Analytics (Roll: A23126552281)
- **GitHub:** [https://github.com/AbhiramKorupalli28/healthcare.git](https://github.com/AbhiramKorupalli28/healthcare.git)
- **License:** MIT License
