import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "tests" / "fixtures" / "sample_2000_rows.csv"   # first 2,000 real rows of the dataset


@pytest.fixture(scope="session")
def spark():
    from src.spark_engine import get_spark
    s = get_spark(app_name="pytest", master="local[2]")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def sample_df(spark):
    from src.spark_engine import read_csv
    return read_csv(spark, str(SAMPLE))
