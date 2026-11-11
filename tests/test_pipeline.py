import pytest

import src.pipeline as pipeline
from src.data_loader import file_fingerprint
from tests.conftest import SAMPLE


def test_pipeline_runs_and_reuses_cached_output(spark, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CLEAN_DIR", tmp_path / "cleaned")
    fp = file_fingerprint(SAMPLE)
    first = pipeline.run_cleaning_pipeline(spark, str(SAMPLE), fingerprint=fp, source_name=SAMPLE.name)
    assert first["status"] == "fresh"
    assert first["before"]["rows"] == 2000
    assert first["codebook_changes"]["labelled"]
    assert first["reconciles"] is True
    assert first["cleaned_rows"] + first["rejected_rows"] == 2000 - first["duplicates_removed"]
    again = pipeline.run_cleaning_pipeline(spark, str(SAMPLE), fingerprint=fp, source_name=SAMPLE.name)
    assert again["status"] == "reused"
    assert again["cleaned_rows"] == first["cleaned_rows"]


def test_local_and_hdfs_outputs_do_not_share_a_cache_key(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CLEAN_DIR", tmp_path)
    local_dir, _ = pipeline._out_dirs("x.csv", "abc", "LOCAL")
    hdfs_dir, _ = pipeline._out_dirs("x.csv", "abc", "HDFS")
    assert local_dir != hdfs_dir
