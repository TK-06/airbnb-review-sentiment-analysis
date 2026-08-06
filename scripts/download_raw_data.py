"""Download and decompress the InsideAirbnb London detailed reviews dataset.

Source: https://insideairbnb.com/get-the-data/ (public, third-party data —
not redistributed in this repo). Run this once before executing
notebooks/1_data_preprocessing.ipynb:

    python scripts/download_raw_data.py
"""
import gzip
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

SNAPSHOT_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/"
    "2026-06-19/data/reviews.csv.gz"
)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
GZ_PATH = RAW_DIR / "reviews.csv.gz"
CSV_PATH = RAW_DIR / "reviews.csv"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def decompress(gz_path: Path, csv_path: Path) -> None:
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    if CSV_PATH.exists():
        print(f"{CSV_PATH} already exists, skipping download.")
        return
    print(f"Downloading {SNAPSHOT_URL} ...")
    download(SNAPSHOT_URL, GZ_PATH)
    print("Decompressing ...")
    decompress(GZ_PATH, CSV_PATH)
    GZ_PATH.unlink()
    print(f"Done: {CSV_PATH}")


if __name__ == "__main__":
    main()
