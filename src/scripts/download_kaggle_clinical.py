"""Download clinical and hospital tables from Kaggle dataset moid1234/health-care-data-set-20-tables."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = "KGAT_f79c1cc438fb216c048d0f7130d4f4b1"
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_ID = "moid1234/health-care-data-set-20-tables"
DEST_DIR = Path("dataset/kaggle_tables")
DEST_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    "STG_EHP_DATASET/STG_EHP__PATN.csv",
    "STG_EHP_DATASET/STG_EHP__VIST.csv",
    "STG_EHP_DATASET/STG_EHP__DIAG/STG_EHP__DIAG_1.csv",
    "STG_EHP_DATASET/STG_EHP__DIAG/STG_EHP__DIAG_2.csv",
    "STG_EHP_DATASET/STG_EHP__BILL.csv",
    "STG_EHP_DATASET/STG_EHP__INSR.csv",
    "STG_EHP_DATASET/STG_EHP__TRTM.csv",
    "STG_EHP_DATASET/STG_EHP__APPT.csv",
    "STG_EHP_DATASET/STG_EHP__MDBL.csv",
    "STG_EHP_DATASET/STG_EHP__MDCN.csv",
]

def download_one(file_name):
    # Check if already downloaded
    target_path = DEST_DIR / file_name
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"Already exists: {file_name} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return file_name, True
    
    api = KaggleApi()
    api.authenticate()
    print(f"Starting download: {file_name}...")
    t0 = time.time()
    try:
        api.dataset_download_file(DATASET_ID, file_name, path=str(DEST_DIR))
        dt = time.time() - t0
        print(f"Finished {file_name} in {dt:.1f}s")
        return file_name, True
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        return file_name, False

def main():
    print(f"Downloading {len(TARGET_FILES)} tables to {DEST_DIR}...")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download_one, f): f for f in TARGET_FILES}
        for fut in as_completed(futures):
            f, ok = fut.result()
            print(f"Status [{f}]: {'SUCCESS' if ok else 'FAILED'}")
    print(f"All downloads finished in {time.time() - t_start:.1f}s")

if __name__ == "__main__":
    main()
