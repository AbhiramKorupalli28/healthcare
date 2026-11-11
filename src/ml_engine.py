"""Machine learning with scikit-learn: Decision Tree, Random Forest and K-Means.

Targets are only the dataset's own labels (Diabetes_Risk, or a Yes/No condition
flag). No risk label is invented. Outputs are statistical patterns in the
data supplied; they are NOT medical diagnoses.

Leakage control: the outcome-derived score and recommendation columns are never
features. Missing values are handled inside the pipeline (median or most-frequent
imputation), fitted on the training split only.
"""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import (FEATURE_ROLES, LEAKAGE_NAME_HINTS, LEAKAGE_ROLES, ML_MAX_ROWS, NON_FEATURE_ROLES,
                    RANDOM_STATE, RF_TREES, TARGET_LEAKAGE, TEST_SIZE)
from src.logger import get_logger

log = get_logger("ml")
DISCLAIMER = ("Statistical pattern from the supplied dataset only. Not a medical diagnosis, "
              "and not clinical advice.")
CLUSTER_ROLES = ["age", "bmi", "glucose", "hba1c", "bp_systolic", "bp_diastolic",
                 "cholesterol", "heart_rate", "sleep_hours", "physical_health_days", "mental_health_days"]
DROP_VALUES = ("Borderline",)   # survey answers that are neither yes nor no; excluded from binary targets


def feature_lists(roles: dict, columns, target: str, df: pd.DataFrame, max_categories: int = 30):
    """Choose model inputs. Returns (numeric, categorical).

    If the dataset has the curated roles (config.FEATURE_ROLES), those are used; otherwise
    every column is a candidate. Never the target, never an ID or weight, never a column
    that restates the outcome (LEAKAGE_ROLES, TARGET_LEAKAGE, name hints).
    """
    excluded = {target, roles.get("patient_id")}
    excluded |= {roles[r] for r in LEAKAGE_ROLES if r in roles}
    excluded |= {roles[r] for r in NON_FEATURE_ROLES if r in roles}
    excluded |= set(TARGET_LEAKAGE.get(target, []))
    curated = [roles[r] for r in FEATURE_ROLES if r in roles and roles[r] != target]
    candidates = curated if curated else list(columns)
    num, cat = [], []
    for c in candidates:
        if c in excluded or c is None or c not in df.columns:
            continue
        if any(h in c for h in LEAKAGE_NAME_HINTS):
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) > 1:
                num.append(c)
        elif 1 < s.nunique(dropna=True) <= max_categories:
            cat.append(c)
    return num, cat


def analysis_columns(roles: dict, all_columns) -> list:
    """The columns the models and clustering need (keeps the Spark-to-pandas sample small)."""
    names = set()
    for r in list(FEATURE_ROLES) + list(LEAKAGE_ROLES) + list(CLUSTER_ROLES) + list(NON_FEATURE_ROLES):
        if r in roles:
            names.add(roles[r])
    for tgt, extra in TARGET_LEAKAGE.items():
        names.add(tgt)
        names.update(extra)
    names.add(roles.get("patient_id", ""))
    return sorted(n for n in names if n and n in set(all_columns))


def _preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _feature_importance(pipe, num, cat) -> pd.DataFrame:
    """Importances of the one-hot columns summed back to the original feature names."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imps = pipe.named_steps["clf"].feature_importances_
    base = num + cat
    agg = {b: 0.0 for b in base}
    for n, v in zip(names, imps):
        raw = n.split("__", 1)[1]
        match = max((b for b in base if raw == b or raw.startswith(b + "_")), key=len, default=None)
        if match:
            agg[match] += float(v)
    out = pd.DataFrame({"feature": list(agg), "importance": list(agg.values())})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def train_classifiers(pdf: pd.DataFrame, target: str, num: list, cat: list, *,
                      max_rows: int = ML_MAX_ROWS, test_size: float = TEST_SIZE,
                      seed: int = RANDOM_STATE, trees: int = RF_TREES) -> dict:
    """Train a Decision Tree and a Random Forest; return metrics, confusion matrices and importances."""
    data = pdf.dropna(subset=[target]).copy()
    data = data[~data[target].isin(DROP_VALUES)]
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=seed)
    X = data[num + cat]
    y_raw = data[target]
    if set(y_raw.dropna().unique()).issubset({0, 1, "0", "1", 0.0, 1.0}):
        y = y_raw.replace({0: "No", 1: "Yes", "0": "No", "1": "Yes", 0.0: "No", 1.0: "Yes"}).astype(str)
    else:
        y = y_raw.astype(str)
    labels = sorted(y.unique())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=trees, class_weight="balanced",
                                                n_jobs=-1, random_state=seed),
    }
    majority = float(y_te.value_counts(normalize=True).max())
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", _preprocessor(num, cat)), ("clf", clf)])
        t = time.time()
        pipe.fit(X_tr, y_tr)
        fit_s = round(time.time() - t, 2)
        pred = pipe.predict(X_te)
        m = {
            "accuracy": accuracy_score(y_te, pred),
            "precision_weighted": precision_score(y_te, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_te, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_te, pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_te, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        }
        if len(labels) == 2:
            pos = "Yes" if "Yes" in labels else labels[1]
            m.update({"precision_positive": precision_score(y_te, pred, pos_label=pos, zero_division=0),
                      "recall_positive": recall_score(y_te, pred, pos_label=pos, zero_division=0),
                      "f1_positive": f1_score(y_te, pred, pos_label=pos, zero_division=0)})
        cm = confusion_matrix(y_te, pred, labels=labels)
        results[name] = {
            "metrics": {k: round(float(v), 4) for k, v in m.items()},
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0),
            "importance": _feature_importance(pipe, num, cat),
            "fit_seconds": fit_s,
            "pipeline": pipe,
        }
        log.info("%s on %s: accuracy %.4f, macro F1 %.4f (fit %.1f s)", name, target,
                 m["accuracy"], m["f1_macro"], fit_s)
    return {
        "target": target, "labels": labels, "features_numeric": num, "features_categorical": cat,
        "rows_used": int(len(data)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "majority_baseline_accuracy": round(majority, 4),
        "results": results, "disclaimer": DISCLAIMER,
    }


def predict_one(bundle: dict, model_name: str, row: dict) -> dict:
    """Prediction interface: one record in, the class and its probabilities out."""
    pipe = bundle["results"][model_name]["pipeline"]
    cols = bundle["features_numeric"] + bundle["features_categorical"]
    X = pd.DataFrame([{c: row.get(c) for c in cols}])
    proba = pipe.predict_proba(X)[0]
    classes = pipe.classes_
    order = np.argsort(-proba)
    return {"prediction": str(classes[order[0]]),
            "probabilities": {str(classes[i]): round(float(proba[i]), 4) for i in order},
            "disclaimer": DISCLAIMER}


def form_defaults(pdf: pd.DataFrame, num: list, cat: list) -> dict:
    """Typical value for each feature (median or most common) to pre-fill the prediction form."""
    out = {}
    for c in num:
        out[c] = float(pdf[c].median()) if pdf[c].notna().any() else 0.0
    for c in cat:
        out[c] = str(pdf[c].mode().iloc[0]) if pdf[c].notna().any() else ""
    return out


def train_kmeans(pdf: pd.DataFrame, features: list, k: int = 3, seed: int = RANDOM_STATE) -> dict:
    """Exploratory clustering on standardised numeric measures. Rows with any missing value are left out."""
    data = pdf[features].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(data.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels, sample_size=min(5000, len(Xs)), random_state=seed)
    profile = data.assign(cluster=labels).groupby("cluster")[features].mean().round(2)
    profile.insert(0, "patients", pd.Series(labels).value_counts().sort_index().values)
    return {"k": k, "features": features, "rows_used": int(len(data)),
            "silhouette": round(float(sil), 4), "profile": profile.reset_index(),
            "limits": ("Clusters only group records that look alike on these measures. They have no labels, "
                       "so they do not show what a cluster means clinically, and the silhouette score is modest "
                       "when clusters overlap.")}
