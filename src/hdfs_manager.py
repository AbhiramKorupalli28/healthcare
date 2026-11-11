"""HDFS access through the WebHDFS REST API, with an explicit local fallback.

Connection rule: the status check asks the NameNode for its root status.
If that answers, every operation goes to HDFS. If not, the same operations run
on a local folder (dataset/hdfs_local_fallback/) and every result is labelled
"LOCAL FALLBACK (not HDFS)" so nobody mistakes local processing for HDFS.
"""
import shutil
from datetime import datetime
from pathlib import Path

import requests

from config import (HDFS_FS_URI, HDFS_INPUT_DIR, HDFS_NAMENODE_URL, HDFS_PROCESSED_DIR,
                    HDFS_TIMEOUT_S, HDFS_USER, LOCAL_HDFS_DIR)
from src.logger import get_logger

log = get_logger("hdfs")
MODE_HDFS = "HDFS"
MODE_LOCAL = "LOCAL FALLBACK (not HDFS)"


class HDFSManager:
    def __init__(self, namenode_url=HDFS_NAMENODE_URL, user=HDFS_USER, fs_uri=HDFS_FS_URI,
                 timeout=HDFS_TIMEOUT_S):
        self.nn = namenode_url.rstrip("/")
        self.user = user
        self.fs_uri = fs_uri
        self.timeout = timeout

    # ---- low level -------------------------------------------------------
    def _url(self, path: str, op: str, **params) -> str:
        extra = "".join(f"&{k}={v}" for k, v in params.items())
        return f"{self.nn}/webhdfs/v1{path}?op={op}&user.name={self.user}{extra}"

    def status(self) -> dict:
        """Connection status. Never raises; reports why HDFS is not reachable."""
        info = {"mode": MODE_LOCAL, "connected": False, "namenode_url": self.nn,
                "fs_uri": self.fs_uri, "live_datanodes": None, "error": None,
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            r = requests.get(self._url("/", "GETFILESTATUS"), timeout=self.timeout)
            if r.status_code == 200:
                info.update(mode=MODE_HDFS, connected=True)
                try:
                    jmx = requests.get(f"{self.nn}/jmx?qry=Hadoop:service=NameNode,name=FSNamesystemState",
                                       timeout=self.timeout).json()
                    info["live_datanodes"] = jmx["beans"][0].get("NumLiveDataNodes")
                except Exception:  # JMX is optional information
                    pass
            else:
                info["error"] = f"NameNode answered HTTP {r.status_code}"
        except requests.RequestException as exc:
            info["error"] = f"NameNode not reachable at {self.nn} ({type(exc).__name__})"
        log.info("HDFS status: %s (%s)", info["mode"], info["error"] or "connected")
        return info

    # ---- operations (HDFS when connected, labelled local fallback otherwise) ---
    def ensure_dirs(self, connected: bool) -> list[str]:
        dirs = [HDFS_INPUT_DIR, HDFS_PROCESSED_DIR]
        if connected:
            for d in dirs:
                r = requests.put(self._url(d, "MKDIRS"), timeout=self.timeout)
                r.raise_for_status()
        else:
            for d in dirs:
                (LOCAL_HDFS_DIR / d.strip("/")).mkdir(parents=True, exist_ok=True)
        log.info("ensured dirs (%s): %s", MODE_HDFS if connected else MODE_LOCAL, dirs)
        return dirs

    def upload(self, local_file, hdfs_dir: str = HDFS_INPUT_DIR, connected: bool = False) -> str:
        """Copy a local file into HDFS (two-step WebHDFS CREATE) or the local fallback."""
        local_file = Path(local_file)
        target = f"{hdfs_dir.rstrip('/')}/{local_file.name}"
        if connected:
            r = requests.put(self._url(target, "CREATE", overwrite="true"), allow_redirects=False,
                             timeout=self.timeout)
            if r.status_code != 307:
                raise RuntimeError(f"HDFS CREATE step answered HTTP {r.status_code}: {r.text[:200]}")
            with open(local_file, "rb") as f:
                r2 = requests.put(r.headers["Location"], data=f, timeout=300)
            if r2.status_code not in (200, 201):
                raise RuntimeError(f"HDFS data write answered HTTP {r2.status_code}")
            log.info("uploaded %s to HDFS %s (%d bytes)", local_file.name, target, local_file.stat().st_size)
            return f"{self.fs_uri}{target}"
        dest = LOCAL_HDFS_DIR / target.strip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_file, dest)
        log.info("copied %s to %s (LOCAL FALLBACK, not HDFS)", local_file.name, dest)
        return str(dest)

    def list_dir(self, hdfs_dir: str, connected: bool) -> list[dict]:
        """Files and folders under hdfs_dir: name, size in bytes, type."""
        rows = []
        if connected:
            r = requests.get(self._url(hdfs_dir, "LISTSTATUS"), timeout=self.timeout)
            r.raise_for_status()
            for st in r.json()["FileStatuses"]["FileStatus"]:
                rows.append({"name": st["pathSuffix"], "type": st["type"].lower(),
                             "size_bytes": st["length"],
                             "modified": datetime.fromtimestamp(st["modificationTime"] / 1000).strftime("%Y-%m-%d %H:%M")})
        else:
            base = LOCAL_HDFS_DIR / hdfs_dir.strip("/")
            if base.exists():
                for p in sorted(base.iterdir()):
                    st = p.stat()
                    rows.append({"name": p.name, "type": "directory" if p.is_dir() else "file",
                                 "size_bytes": st.st_size if p.is_file() else 0,
                                 "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")})
        return rows

    def download(self, hdfs_path: str, dest, connected: bool) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if connected:
            with requests.get(self._url(hdfs_path, "OPEN"), stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        else:
            shutil.copyfile(LOCAL_HDFS_DIR / hdfs_path.strip("/"), dest)
        return dest

    def storage_root(self, connected: bool) -> str:
        return f"{self.fs_uri}" if connected else str(LOCAL_HDFS_DIR)


def get_hdfs() -> tuple[HDFSManager, dict]:
    """Manager plus its current status. The status decides the mode for all later calls."""
    mgr = HDFSManager()
    return mgr, mgr.status()
