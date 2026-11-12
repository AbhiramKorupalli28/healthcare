"""Build the real combined dataset used by the project.

Source: CDC Behavioral Risk Factor Surveillance System (BRFSS), annual survey files
for 2021, 2022 and 2023 (public, US federal government work, no licence fee).
  https://www.cdc.gov/brfss/annual_data/annual_2022.html  (and 2021, 2023)

What it does, one year at a time (to keep disk use low):
  1. download the official SAS transport (.XPT) zip for the year
  2. read the XPT in chunks with pandas (no record is generated or altered)
  3. keep only the variables that all three years share, plus a survey_year column
  4. append to dataset/brfss_2021_2023_combined.csv and delete the XPT file
  5. write dataset/PROVENANCE.json with the real row and column counts

Run:  python scripts/build_dataset.py
"""
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
WORK = DATA / "_download"
OUT = DATA / "brfss_2021_2023_combined.csv"
YEARS = [2021, 2022, 2023]
URL = "https://www.cdc.gov/brfss/annual_data/{y}/files/LLCP{y}XPT.zip"
CHUNK = 50_000


def download(year: int) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    zpath = WORK / f"LLCP{year}XPT.zip"
    if not zpath.exists():
        print(f"downloading {URL.format(y=year)}", flush=True)
        # curl: the CDC site refuses some default HTTP-client agents with 403
        subprocess.run(["curl", "-sSL", "--fail", "-m", "900", "-o", str(zpath), URL.format(y=year)], check=True)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.strip().upper().endswith(".XPT"))
        z.extract(name, WORK)
    zpath.unlink()
    # CDC names the member with a trailing space; store it under a clean name
    out = WORK / f"LLCP{year}.XPT"
    (WORK / name).rename(out)
    return out


def xpt_columns(path: Path) -> list[str]:
    reader = pd.read_sas(path, format="xport", chunksize=1, encoding="utf-8")
    return list(next(reader).columns)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    # first pass: the variables that every year has (read headers only)
    headers = {}
    for y in YEARS:
        xpt = download(y)
        headers[y] = xpt_columns(xpt)
        xpt.unlink()
        print(f"{y}: {len(headers[y])} variables", flush=True)
    common = [c for c in headers[YEARS[0]] if all(c in headers[y] for y in YEARS[1:])]
    print(f"variables shared by all years: {len(common)}", flush=True)

    if OUT.exists():
        OUT.unlink()
    rows_by_year, total = {}, 0
    for y in YEARS:
        xpt = download(y)
        n = 0
        first = not OUT.exists()
        for chunk in pd.read_sas(xpt, format="xport", chunksize=CHUNK, encoding="utf-8"):
            part = chunk[common].copy()
            part.insert(0, "survey_year", y)
            part.to_csv(OUT, mode="a", header=first, index=False)
            first = False
            n += len(part)
        xpt.unlink()
        rows_by_year[y] = n
        total += n
        print(f"{y}: {n:,} respondents appended", flush=True)

    size = OUT.stat().st_size
    prov = {
        "name": "BRFSS annual survey files combined, 2021-2023 (shared variables)",
        "source": "CDC BRFSS: " + ", ".join(URL.format(y=y) for y in YEARS),
        "licence": "US federal government public data (CDC)",
        "file": OUT.name,
        "size_bytes": size,
        "size_mb": round(size / 1_000_000, 2),
        "rows": total,
        "rows_by_year": rows_by_year,
        "columns": len(common) + 1,
        "variables_per_year": {str(y): len(headers[y]) for y in YEARS},
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic": False,
        "note": "Survey answers as recorded by CDC; not weighted in this project.",
    }
    (DATA / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    shutil.rmtree(WORK, ignore_errors=True)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    sys.exit(main())
