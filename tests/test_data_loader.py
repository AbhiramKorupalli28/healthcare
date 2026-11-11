import pandas as pd

from src.data_loader import detect_roles, file_fingerprint, preview_pandas, to_snake
from tests.conftest import SAMPLE


def test_to_snake_standardises_names():
    assert to_snake("Blood_Pressure Systolic") == "blood_pressure_systolic"
    assert to_snake("BMI") == "bmi"
    assert to_snake("  ") == "column"


def test_detect_roles_uses_whatever_columns_exist():
    roles = detect_roles(["Blood_Glucose", "Age", "Gender", "Some_Other_Column"])
    assert roles["glucose"] == "blood_glucose"
    assert roles["age"] == "age"
    assert "bmi" not in roles          # absent columns are not invented


def test_fingerprint_changes_when_file_changes(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x,y\n1,2\n")
    first = file_fingerprint(f)
    assert first == file_fingerprint(f)
    f.write_text("x,y\n1,2\n3,4\n")
    assert file_fingerprint(f) != first


def test_preview_reads_real_rows():
    p = preview_pandas(SAMPLE, 5)
    assert isinstance(p, pd.DataFrame) and len(p) == 5
    assert "_AGE80" in p.columns
