# Healthcare Big Data Analytics: Final Project Report

**Author:** Abhiram Korupalli  
**Course:** B.Tech Big Data Analytics (Roll: A23126552281)  
**Repository:** [https://github.com/AbhiramKorupalli28/healthcare.git](https://github.com/AbhiramKorupalli28/healthcare.git)

---

## 1. Executive Summary

This project establishes an end-to-end Big Data analytics pipeline designed to ingest, process, query, and model large-scale healthcare data (1.6+ million patient records, >1.5 GB in-memory Spark partitions). Using Apache Spark, Spark SQL, Hadoop HDFS, and scikit-learn, the system extracts population-level health trends, evaluates chronic condition patterns, and predicts disease outcomes with up to 96.9% accuracy.

---

## 2. Dataset Overview

### A. Kaggle Enterprise Hospital Multi-Table Dataset
- **Source:** [Kaggle Healthcare Dataset (20 Tables)](https://www.kaggle.com/datasets/moid1234/health-care-data-set-20-tables)
- **Scale:** 1,602,174 records, 32 clinical & administrative columns (~405 MB CSV, >1.5 GB in-memory Spark execution).
- **Core Tables:**
  - `STG_EHP__PATN`: 351,765 Patient demographics (DOB, Gender, Blood Type, Marital Status)
  - `STG_EHP__VIST`: 917,331 Hospital admissions (Entry/Exit, Length of Stay, Room Type)
  - `STG_EHP__DIAG_1` & `2`: 1,602,174 Diagnoses, ICD codes, and treatments
  - `STG_EHP__BILL`: 917,331 Billing records and payment status
  - `STG_EHP__INSR`: Health insurance providers and coverage details

### B. CDC BRFSS Multi-Year Population Survey
- **Source:** US Centers for Disease Control and Prevention (2021, 2022, 2023 Combined)
- **Scale:** 1,317,148 records, 194 variables (821.08 MB).

---

## 3. Data Cleaning & Validation Results

The PySpark distributed cleaning pipeline executed across all 1,602,174 raw records:
- **Duplicates Removed:** 846 exact duplicate records
- **Valid Clean Records:** 1,601,328 records (100% of validated rows)
- **Rejected Records:** 0 records outside physiological thresholds
- **Storage:** Saved as partitioned, Snappy-compressed Parquet for sub-second analytical query access.

---

## 4. Spark SQL Key Analytical Findings

| Query / Metric | Primary Finding |
|---|---|
| **Total Clean Records** | 1,601,328 hospital patient encounters |
| **Hypertension Prevalence** | 269,527 patients (16.8%) |
| **Diabetes Prevalence** | 80,681 patients |
| **Heart Disease Prevalence** | 80,320 patients |
| **Cancer Prevalence** | 40,191 patients |
| **Kidney Disease Prevalence** | 39,948 patients |
| **Blood Pressure Stratification** | Stage 2 Hypertension: 46.8%, Stage 1: 31.2%, Elevated: 14.5%, Normal: 7.5% |
| **Risk Classification** | High Risk: 1,578,206 (multimorbidity / surgery), Moderate Risk: 20,344, Low Risk: 2,778 |

---

## 5. Machine Learning Evaluation

Models were trained using an 80/20 train/test split with balanced class weighting on 50,000 stratified samples:

| Target Condition | Majority Baseline | Decision Tree Accuracy | Random Forest Accuracy | Random Forest Precision | Random Forest Recall | Random Forest Macro F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heart Disease** | 94.98% | 89.98% | **96.92%** | 0.9692 | 0.9692 | **0.8444** |
| **Diabetes** | 94.96% | 49.53% | **92.28%** | 0.9228 | 0.9228 | **0.6620** |
| **Hypertension** | 83.18% | 77.45% | **89.72%** | 0.8972 | 0.8972 | **0.8290** |
| **Risk Category** | 98.60% | 10.46% | **51.88%** | 0.9856 | 0.5188 | 0.2396 |

**Key Takeaways:**
- **Random Forest** substantially outperformed Decision Trees across all diagnostic targets.
- For **Heart Disease**, Random Forest achieved **96.92% accuracy** and a Macro F1 of **0.8444**, beating the majority baseline while maintaining high recall on rare positive cases.
- For **Diabetes**, Random Forest achieved **92.28% accuracy** (Macro F1 of 0.6620) compared to Decision Tree's 49.53%.

---

## 6. System Architecture & Dashboard

The application runs a multi-page interactive Streamlit dashboard featuring:
1. **Overview:** High-level metrics, disease distribution, and risk cohort breakdowns.
2. **Dataset Explorer:** Dynamic CSV uploads up to 4 GB with 100-row previews and Spark schema inspection.
3. **Disease Analytics:** Multi-variable slice-and-dice filtering by age group, gender, and clinical vitals.
4. **Risk Analysis:** Interactive ML risk score calculator for real-time patient assessment.
5. **Big Data Processing:** Cluster telemetry, WebHDFS status, and PySpark job monitoring.
6. **Reports:** Exportable CSV summaries and printable reports.
