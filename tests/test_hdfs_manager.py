import pytest
import requests

from src import hdfs_manager
from src.hdfs_manager import HDFSManager, MODE_HDFS, MODE_LOCAL


def test_unreachable_namenode_reports_local_fallback_not_hdfs():
    st = HDFSManager(namenode_url="http://127.0.0.1:9", timeout=1).status()
    assert st["connected"] is False
    assert st["mode"] == MODE_LOCAL
    assert "not reachable" in st["error"]


def test_local_fallback_upload_list_download(tmp_path, monkeypatch):
    monkeypatch.setattr(hdfs_manager, "LOCAL_HDFS_DIR", tmp_path / "fallback")
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,2\n")
    mgr = HDFSManager(namenode_url="http://127.0.0.1:9", timeout=1)
    mgr.ensure_dirs(connected=False)
    assert (tmp_path / "fallback" / "healthcare" / "input").is_dir()
    mgr.upload(src, "/healthcare/input", connected=False)
    rows = mgr.list_dir("/healthcare/input", connected=False)
    assert [r["name"] for r in rows] == ["data.csv"]
    out = mgr.download("/healthcare/input/data.csv", tmp_path / "back.csv", connected=False)
    assert out.read_text() == "a,b\n1,2\n"


@pytest.mark.skipif(HDFSManager().status()["connected"] is False, reason="real HDFS not running")
def test_live_hdfs_roundtrip(tmp_path):
    mgr = HDFSManager()
    mgr.ensure_dirs(connected=True)
    src = tmp_path / "probe.csv"
    src.write_text("x\n1\n")
    mgr.upload(src, "/healthcare/input", connected=True)
    names = [r["name"] for r in mgr.list_dir("/healthcare/input", connected=True)]
    assert "probe.csv" in names
    back = mgr.download("/healthcare/input/probe.csv", tmp_path / "probe_back.csv", connected=True)
    assert back.read_text() == "x\n1\n"
