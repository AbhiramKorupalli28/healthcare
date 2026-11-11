# Healthcare Analytics for Disease Pattern Detection

A Big Data analytics project built for a B.Tech Big Data Analytics mini case study (roll A23126552281). It processes a large public health survey with Apache Spark (PySpark, Spark SQL), stores data in Hadoop HDFS, trains scikit-learn models, and shows the results in a Streamlit dashboard.

Every number in this README and in the dashboard comes from a real run on the dataset below. The run outputs are in `output/run_results_local.json` and `output/run_results_hdfs.json`.

---

## 1. Problem statement

Health records describe large populations, but the useful patterns (which conditions are common, how they change with age, sex and body weight, and which groups are at higher risk) are hidden in files too big for a spreadsheet. The question is how to store, clean, query and model such a file reliably, and show the results to people who need to read them.

## 2. Objectives

1. Store a large healthcare dataset in distributed storage (HDFS) and read it back.
2. Clean and validate it with distributed processing (PySpark), and report before/after statistics.
3. Answer population-level questions with Spark SQL (disease counts, age and gender patterns, BMI, blood pressure, missing values, risk categories).
4. Train and evaluate classification models (Decision Tree, Random Forest) and an exploratory K-Means clustering, with honest metrics.
5. Present everything in an interactive Streamlit dashboard, with reports that can be downloaded.

## 3. Technologies

| Area | Used |
|---|---|
| Language | Python 3.11 |
| Distributed storage | Hadoop HDFS 3.3.6 (single node, pseudo-distributed, tested on this machine) |
| Distributed processing | Apache Spark 3.5.3 through PySpark 3.5.3 (local mode, all cores) |
| Analytical queries | Spark SQL (temporary view `healthcare_patients`) |
| Data manipulation | Pandas, NumPy (only on Spark-aggregated results and on a Spark sample for modelling) |
| Machine learning | scikit-learn 1.9.1: DecisionTreeClassifier, RandomForestClassifier, KMeans |
| Charts | Plotly (dashboard), Matplotlib (visual report PNGs) |
| Dashboard | Streamlit 1.67.0 (no separate HTML/CSS/JavaScript/React/Node frontend) |
| Formats | CSV (input, downloads), Parquet (cleaned and rejected records) |
| Tests | pytest 9.2.0 |

## 4. System architecture

```
  CDC survey files (2021, 2022, 2023)  ──scripts/build_dataset.py──▶  dataset/brfss_2021_2023_combined.csv
                                                                       │
                              upload to HDFS  (src/hdfs_manager.py, WebHDFS REST)
                                                                       │
                       ┌──────────────── HDFS /healthcare/input/ ◀─────┘   (or local copy when HDFS is down)
                       │
   Spark reads CSV  ── src/codebook.py  (CDC codes → readable names and labels)
                    ── src/preprocessing.py  (names, types, duplicates, validity rules, derived categories)
                    ── src/pipeline.py  (run once per file; output cached by file fingerprint)
                       │
          cleaned.parquet + rejected.parquet  ──▶  HDFS /healthcare/processed/   (or output/cleaned/)
                       │
     ┌─────────────────┼────────────────────────────┬────────────────────────────┐
     ▼                 ▼                            ▼                            ▼
 Spark SQL          Spark aggregations          Spark sample (50,000)        Dashboard
 src/sql_analytics  src/spark_engine,           → scikit-learn               app.py + pages/
 (9 queries)        src/analytics               src/ml_engine               (6 pages, Plotly)
```

Modules:

| File | Role |
|---|---|
| `app.py` | Streamlit entry point: navigation, sidebar, theme |
| `config.py` | Paths, HDFS and Spark settings, column roles and aliases, validity ranges, theme colours |
| `pages/*.py` | The six dashboard pages (overview, dataset explorer, disease analytics, risk analysis, big data processing, reports) |
| `src/data_loader.py` | Dataset discovery, file fingerprints, dynamic column detection, light previews |
| `src/hdfs_manager.py` | HDFS status, `/healthcare/input` and `/healthcare/processed`, upload, list, read; labelled local fallback |
| `src/codebook.py` | CDC codebook: renames, code-to-label maps, special codes |
| `src/preprocessing.py` | Standardised names, duplicates, types, invalid records, consistency checks, derived categories (PySpark) |
| `src/pipeline.py` | End-to-end cleaning run, cached per file fingerprint, with run statistics and a processing log |
| `src/spark_engine.py` | Spark session and distributed aggregations |
| `src/sql_analytics.py` | The nine Spark SQL queries, built from detected columns |
| `src/analytics.py` | Page-level analyses (Spark) |
| `src/ml_engine.py` | Feature selection, leakage control, Decision Tree, Random Forest, metrics, K-Means, prediction |
| `src/visualization.py` | Plotly charts (dark theme) and matplotlib PNGs |
| `src/logger.py` | Processing log at `output/logs/processing.log` |
| `run_analysis.py` | Command-line run of the whole analysis (writes `output/run_results_*.json`) |
| `scripts/build_dataset.py` | Downloads the three CDC files and builds the combined CSV |
| `tests/` | 23 unit and integration tests (real records from the dataset) |

## 5. Dataset

**Name:** CDC Behavioral Risk Factor Surveillance System (BRFSS), annual survey files for 2021, 2022 and 2023, combined on the variables all three years share.

**Source:** US Centers for Disease Control and Prevention (CDC):
- https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip
- https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip
- https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip

**Licence:** US federal government work (public data).

**Size and shape (from the real build, `dataset/PROVENANCE.json`):**

| Item | Value |
|---|---|
| Combined file | `brfss_2021_2023_combined.csv` |
| Size | 821,075,755 bytes = 821.08 MB (0.82 GB) |
| Rows (respondents) | 1,317,148 |
| Rows by year | 2021: 438,693; 2022: 445,132; 2023: 433,323 |
| Columns | 194 (193 variables that all three years share, plus `survey_year`) |
| Variables per year before sharing | 2021: 303; 2022: 328; 2023: 350 |

**Not 1 GB.** The brief asked for at least about 1 GB. This file is 0.82 GB. Keeping only the variables all three years share is what brings it below 1 GB. Keeping every variable, with blanks where a year lacks one, would be larger.

**What the survey contains (used by this project):** age (`_AGE80`, 18 to 80, top-coded), sex (`SEXVAR`), BMI (`_BMI5`, two implied decimals), CDC's BMI category (`_BMI5CAT`), diabetes (`DIABETE4`), heart disease (`CVDCRHD4`), heart attack (`CVDINFR4`), stroke (`CVDSTRK3`), asthma (`ASTHMA3`), COPD (`CHCCOPD3`), general health, physical and mental health days, physical activity, smoking status, and the survey weight.

**What it does not contain (stated plainly):**
- No glucose or cholesterol measurements, and no blood-pressure readings. This is a survey of self-reported answers, not a laboratory record.
- The hypertension question (`BPHIGH6`), the high-cholesterol question (`TOLDHI3`), the kidney-disease question, the cancer question and the sleep-hours question are not among the 193 variables all three years share, so they are not in this file. Two queries that need them report "not in dataset" and are not estimated.
- The file has 1,317,148 respondents, so counts are sample counts, not population rates. The survey weight (`_LLCPWT`) is kept but not applied.

**Missing values:** 118,823,538 of 255,526,712 cells (46.50%) are blank in the raw file. Most of this is in variables that only some respondents were asked (for example the sub-module questions). They are kept as null in the cleaned data, and the models impute them inside their pipeline.

## 6. Installation

### 6.1 Linux (tested on this project's machine: Debian 12, 16 CPUs, Java 17, Python 3.11)

```bash
# Java 17 (Spark 3.5 supports Java 8, 11 and 17)
sudo apt-get install -y openjdk-17-jdk-headless
java -version

# Python environment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Build the dataset (downloads the three CDC files, several minutes)
python scripts/build_dataset.py

# Hadoop 3.3.6 (only for real HDFS; see section 8)
```

### 6.2 Windows (written from the steps for this stack; the Windows steps were not run on this machine)

1. **Python 3.11** from python.org. Tick "Add python.exe to PATH". Check: `python --version`.
2. **Java 17** (Eclipse Temurin 17 JDK). Set `JAVA_HOME` to the install folder (for example `C:\Program Files\Eclipse Adoptium\jdk-17...`) and add `%JAVA_HOME%\bin` to `PATH`. Check: `java -version`.
3. **Project environment:**
   ```
   cd healthcare-bigdata
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -c "import pyspark; print(pyspark.__version__)"
   ```
   PySpark ships with Apache Spark 3.5.3, so no separate Spark download is needed for local mode. Check: `python -c "from src.spark_engine import get_spark; print(get_spark().version)"`.
4. **Hadoop 3.3.6 (only for real HDFS):** download the `hadoop-3.3.6` binary, and the matching `winutils.exe` and `hadoop.dll` for Windows. Put them in `C:\hadoop\bin`. Set `HADOOP_HOME=C:\hadoop` and add `%HADOOP_HOME%\bin` to `PATH`. Check: `hadoop version`.
5. Configure Hadoop as in section 8, then format and start the NameNode and DataNode.

The application falls back to local storage when HDFS is not reachable, and says so on every page.

## 7. Implementation procedure

1. **Dataset (`scripts/build_dataset.py`).** Each year's SAS transport file is downloaded from CDC and read in chunks with pandas. Only the variables shared by all three years are kept, and a `survey_year` column is added. Nothing is generated or altered.
2. **Column detection (`src/data_loader.py`, `config.py`).** Columns are matched to roles (age, gender, BMI, diseases, and so on) by a table of alternative names, after standardisation to snake_case. A role with no matching column is simply absent, and anything that needs it says so.
3. **Codebook (`src/codebook.py`).** CDC variable names are renamed to readable names and their codes are labelled (for example `SEXVAR` 1 becomes "Male"). Codes 7, 9, 77 and 99 (don't know, refused, not asked) become null. BMI is divided by 100 because CDC stores it with two implied decimals.
4. **Spark cleaning (`src/preprocessing.py`, `src/pipeline.py`).** Steps, each counted for the before/after statistics:
   - standardise the names;
   - profile the raw data (rows, columns, missing cells, duplicates);
   - remove exact duplicate rows;
   - convert types (text trimmed, blank text to null, numeric roles to double);
   - flag invalid records against plausible ranges (age 18 to 120, BMI 10 to 100, physical and mental health days 0 to 30, sleep 1 to 24). Invalid records go to `rejected.parquet`;
   - check consistency (where the columns exist);
   - add derived categories: age group (18–29, 30–39, …, 70+) and WHO BMI class, both null when the input is missing;
   - write `cleaned.parquet` and `rejected.parquet`, and `run_stats.json`.
   The run reconciles: raw rows minus duplicates minus rejected equals cleaned rows.
5. **Spark SQL (`src/sql_analytics.py`).** The cleaned data is registered as a temporary view, and nine queries run on it (section 9).
6. **Machine learning (`src/ml_engine.py`).** Targets are only the dataset's own columns (section 10).
7. **Dashboard (`app.py`, `pages/`).** Six pages. Each page calls cached Spark aggregations, so a new upload or a changed file produces new results and an unchanged file reuses the old ones.

## 8. HDFS

### 8.1 Real HDFS used for these results

Hadoop 3.3.6 was installed on the test machine and run as a single-node pseudo-distributed cluster (replication 1, Java 17). The configuration was:

`etc/hadoop/core-site.xml`
```xml
<configuration>
  <property><name>fs.defaultFS</name><value>hdfs://localhost:9000</value></property>
  <property><name>hadoop.tmp.dir</name><value>/home/user/hdfs/tmp</value></property>
</configuration>
```

`etc/hadoop/hdfs-site.xml`
```xml
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file:///home/user/hdfs/namenode</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file:///home/user/hdfs/datanode</value></property>
  <property><name>dfs.permissions.enabled</name><value>false</value></property>
  <property><name>dfs.namenode.http-address</name><value>localhost:9870</value></property>
  <property><name>dfs.datanode.hostname</name><value>localhost</value></property>
</configuration>
```

`etc/hadoop/hadoop-env.sh` (appended): `JAVA_HOME` set to the Java 17 path and `HADOOP_OPTS` with the `--add-opens` flags that Hadoop 3.3 needs on Java 17.

Start (Linux):
```bash
export HADOOP_HOME=/path/to/hadoop-3.3.6
export PATH=$HADOOP_HOME/bin:$PATH
hdfs namenode -format -force -nonInteractive     # first time only
hdfs --daemon start namenode
hdfs --daemon start datanode
jps                                              # NameNode and DataNode should be listed
```

Web UI: http://localhost:9870 (NameNode). The application reads its status from `http://localhost:9870/webhdfs/v1/?op=GETFILESTATUS`.

On Windows, run `hdfs namenode` and `hdfs datanode` in two terminals (daemon mode is not used there). This was not tested on Windows.

### 8.2 HDFS commands

```bash
hdfs dfs -mkdir -p /healthcare/input /healthcare/processed
hdfs dfs -put dataset/brfss_2021_2023_combined.csv /healthcare/input/
hdfs dfs -ls /healthcare/input /healthcare/processed
hdfs dfs -du -h /healthcare/input
hdfs dfs -get /healthcare/input/brfss_2021_2023_combined.csv ./copy.csv
hdfs dfsadmin -report | head -20
```

The dashboard's Big Data Processing page runs the same operations through WebHDFS: create the two directories, upload the active dataset, list both directories, and re-run the Spark cleaning from the HDFS copy.

### 8.3 Local fallback

If the NameNode does not answer, `src/hdfs_manager.py` uses `dataset/hdfs_local_fallback/healthcare/…` instead. Every page then shows the storage mode as **LOCAL FALLBACK (not HDFS)**, and the Big Data Processing page explains why.

## 9. Spark processing and Spark SQL

Run the whole analysis from the command line:

```bash
python run_analysis.py                 # local files
python run_analysis.py --hdfs          # reads the HDFS copy, writes cleaned Parquet to /healthcare/processed/
python run_analysis.py --force         # ignore cached cleaning output
```

Spark runs in local mode (`SPARK_MASTER=local[*]`) with a 3 GB driver. The same code runs on a cluster by changing `SPARK_MASTER`, `HDFS_FS_URI` and the memory settings in `config.py` or the environment.

The nine Spark SQL queries (run on the cleaned view `healthcare_patients`):

1. Total patient records.
2. Disease-wise patient count (Yes out of known answers).
3. Age-group disease distribution (prevalence by age group).
4. Gender-wise disease distribution.
5. Average BMI by disease.
6. Average glucose by disease: **not available in this dataset** (no glucose measurement).
7. Blood pressure: **not available in this dataset** (no readings and no hypertension question in all three years).
8. Missing-value analysis, one scan, one row per column.
9. Risk-category distribution (CDC BMI category, with "Missing" shown).

Results (local run; the HDFS run gives the same counts):

**Query 1.** Total records: 1,317,148 after cleaning, from 1,317,148 raw rows.

**Query 2.** Disease counts (positives out of records with a known Yes/No answer):

| Condition | Yes | Known answers | Prevalence |
|---|---|---|---|
| asthma | 192,562 | 1,311,928 | 14.68% |
| diabetes | 189,457 | 1,283,227 | 14.76% |
| copd | 102,922 | 1,310,756 | 7.85% |
| heart_disease | 72,896 | 1,304,237 | 5.59% |
| heart_attack | 71,390 | 1,308,860 | 5.45% |
| stroke | 54,802 | 1,312,688 | 4.17% |

**Query 4.** By gender: female 700,317 records (diabetes 14.49%, heart disease 4.33%, asthma 17.07%), male 616,831 records (diabetes 14.26%, heart disease 6.91%, asthma 11.84%). Full table in `output/summaries/4_gender_distribution.csv`.

**Query 5.** Average BMI by diabetes status: Yes 31.61 (n = 189,457), No 27.9 (n = 1,093,770), Borderline 31.2.

**Query 3.** Age groups (records): 

| Age group | Records |
|---|---|
| 18-29 | 144,138 |
| 30-39 | 161,303 |
| 40-49 | 176,541 |
| 50-59 | 221,852 |
| 60-69 | 272,709 |
| 70+ | 340,605 |

**Query 9.** Risk category (CDC BMI category):

| Risk category | Records |
|---|---|
| Overweight | 418,370 |
| Obese | 393,788 |
| Normal weight | 348,965 |
| Missing | 136,193 |
| Underweight | 19,832 |

**Query 8.** Missing values: 196 columns (194 raw plus the two derived categories) in `output/summaries/8_missing_values.csv`.

**Spark timings (local run, 1,317,148 rows):** read CSV 6.95 s, profile raw 17.44 s, clean 14.23 s, write Parquet 6.55 s, profile cleaned 14.06 s; cleaning in total 59.26 s; whole analysis 82.49 s. HDFS run: cleaning 59.63 s; whole analysis 83.83 s.

Spark never loads the whole table into pandas for analysis. Only aggregated results and one Spark sample of 50,000 records (for modelling) reach pandas.

## 10. Machine learning methodology

- **Targets.** Only the dataset's own columns: `risk_category` (CDC BMI category, four classes), `diabetes` (Yes/No, Borderline excluded), `heart_disease` (Yes/No). Nothing is invented or relabelled.
- **Sample.** Spark samples up to 50,000 records of the modelling columns (seed 42). The models see this sample, not all 1.3 million records.
- **Features.** A curated list in `config.py`: age, sex, BMI, general health, physical and mental health days, physical activity, smoking status, and the other condition flags. The target is never a feature.
- **Leakage control.** Columns that restate the target are excluded: the BMI and BMI-derived flag for the BMI-category target, and the risk label for every other target.
- **Missing values.** Imputed inside the scikit-learn pipeline (median for numbers, most frequent for categories), fitted on the training split only.
- **Split.** Stratified 80/20 train and test split, random state 42.
- **Models.** Decision Tree (max depth 6) and Random Forest (200 trees), both with balanced class weights.
- **Metrics.** Accuracy, precision, recall, F1 (weighted and macro), a confusion matrix, a per-class report, and feature importance (Random Forest, one-hot columns summed back to the original feature).
- **Baseline.** The accuracy of always predicting the most common class is reported next to every model, because the classes are imbalanced.
- **Prediction interface.** On the Risk Analysis page: a form with typical values pre-filled, the predicted class and its probabilities, and the disclaimer below.

**Important:** these models describe patterns in survey answers. They are not medical diagnoses and not clinical advice.

Results (local run, sample of 50,000 records; the HDFS run used a slightly different sample, and its numbers are in `output/run_results_hdfs.json`):

| Target | Majority-class accuracy | Decision Tree accuracy | Decision Tree macro F1 | Random Forest accuracy | Random Forest precision (weighted) | Random Forest recall (weighted) | Random Forest F1 (weighted) | Random Forest macro F1 |
|---|---|---|---|---|---|---|---|---|
| `risk_category` | 0.3537 | 0.3746 | 0.3204 | 0.3928 | 0.4094 | 0.3928 | 0.4000 | 0.3159 |
| `diabetes` | 0.8539 | 0.7030 | 0.6077 | 0.7939 | 0.8093 | 0.7939 | 0.8010 | 0.6161 |
| `heart_disease` | 0.9450 | 0.7850 | 0.5811 | 0.9335 | 0.9355 | 0.9335 | 0.9345 | 0.6892 |

Read these carefully. For `diabetes` and `heart_disease` the most-common-class baseline is higher than either model's accuracy, because the classes are very unbalanced (for example only 5.6% of records have a heart-disease Yes). The models trade accuracy for recall on the rare class, which the balanced weighting asks for. Macro F1 is the more honest summary here. For `risk_category` the models are only slightly better than the baseline.

Feature importance (Random Forest, `diabetes` target) and the confusion matrices are on the Risk Analysis page.

**K-Means (exploratory, no labels).** Three clusters on age, BMI, physical health days and mental health days, standardised, 43,272 records with no missing value in these measures (local run). Silhouette score 0.3219 (local) and 0.324 (HDFS). A modest score means the clusters overlap. The clusters group similar respondents; they do not say what a cluster means clinically.

## 11. Dashboard

Run it with `streamlit run app.py`, then open http://localhost:8501.

| Page | What it shows |
|---|---|
| Overview | Four metric cards (valid records, disease categories, dataset size, missing cells with duplicates and rejected records); disease distribution; age distribution; risk-category distribution; a summary computed from the run |
| Dataset Explorer | CSV upload (saved and then used as the active dataset); 100-row preview; column filter; Spark schema; missing values per column; duplicate count |
| Disease Analytics | Filters for gender and age group; disease frequency; age-wise and gender-wise prevalence; clinical measurement summary (BMI) and averages by disease status |
| Risk Analysis | Risk-category distribution; train Decision Tree and Random Forest on a chosen target; metrics, baseline, confusion matrix, feature importance; K-Means; prediction form |
| Big Data Processing | HDFS status (HDFS or LOCAL FALLBACK, with the reason); create directories; upload; list `/healthcare/input` and `/healthcare/processed`; Spark status; processing statistics; storage; processing log |
| Reports | Downloads (sample CSV, disease frequency, every Spark SQL result as CSV); visual report PNGs (zip); key findings computed from the run |

Charts update when the active dataset changes. Long steps show a progress bar. Cleaned results are cached per file fingerprint, and each analysis is cached per dataset version.

## 12. Execution

```bash
pip install -r requirements.txt
python scripts/build_dataset.py       # once: real CDC files, several minutes
python run_analysis.py                # cleaning, Spark SQL, models, K-Means (about 1.5 minutes)
streamlit run app.py                  # dashboard
python -m pytest -q tests             # tests (about 1 minute; they use real records)
```

With HDFS running: `python run_analysis.py --hdfs`.

## 13. Expected results

These are the values the code produced on the test machine. Your numbers should match when you use the same file, the same versions and the same seed.

- Records: 1,317,148 valid of 1,317,148 raw; 0 duplicates; 0 rejected; reconciles.
- Missing cells in the raw file: 46.50%.
- Most common condition by count: asthma (192,562 of 1,311,928 known answers, 14.68%). Highest prevalence among known answers: diabetes (189,457 of 1,283,227, 14.76%).
- Models: see the table in section 10. For `diabetes` and `heart_disease` no model beats the majority-class accuracy; macro F1 is the more useful summary.
- Tests: 23 passed.

## 14. Tests

`tests/` contains 23 tests: the data loader (names, roles, fingerprints, preview), HDFS (fallback, and a live round trip when HDFS is running), preprocessing (codebook, reconciliation, invalid records, derived categories), Spark SQL and aggregation results (checked against pandas), the models (confusion matrix totals, probabilities, leakage exclusions), K-Means, and the pipeline (fresh and reused runs). The fixture `tests/fixtures/sample_2000_rows.csv` is the first 2,000 real rows of the combined file.

## 15. Limits and what is not done

- The dataset is a survey (self-reported answers), not clinical records. It has no glucose, cholesterol or blood-pressure measurements, and some risk questions are not in all three years.
- The combined file is 0.82 GB, not 1 GB.
- The models are classroom-level baselines. Their accuracy does not beat the majority baseline for two of three targets, and this is reported as it is.
- Windows steps were written for this stack but not run on Windows.
- Spark runs in local mode on one machine. A multi-node cluster was not built.
- Survey weights are not applied, so counts are sample counts.
- The CSV upload accepts any CSV, but the curated feature list and the dataset-specific codebook apply only to the CDC variables.

## 16. Project files

```
healthcare-bigdata/
  app.py  config.py  requirements.txt  README.md  run_analysis.py
  .streamlit/config.toml
  pages/   overview.py  dataset_explorer.py  disease_analytics.py  risk_analysis.py  bigdata_processing.py  reports.py
  src/     data_loader.py  hdfs_manager.py  codebook.py  preprocessing.py  pipeline.py  spark_engine.py
           sql_analytics.py  analytics.py  ml_engine.py  visualization.py  ui.py  app_state.py  logger.py
  scripts/ build_dataset.py
  dataset/ PROVENANCE.json  (the combined CSV is built by scripts/build_dataset.py; not in the zip)
  output/  run_results_local.json  run_results_hdfs.json  summaries/  logs/processing.log
  reports/ visual report PNGs (generated from the dashboard)
  tests/   test_*.py  fixtures/sample_2000_rows.csv  conftest.py
```
