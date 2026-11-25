"""Download the UCI Student Performance dataset (CC BY 4.0)."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

print("Downloading the official UCI Student Performance dataset...")
try:
    with urlopen(URL, timeout=30) as response:
        content = response.read()
    with ZipFile(BytesIO(content)) as archive:
        for name in ("student-mat.csv", "student-por.csv"):
            matches = [n for n in archive.namelist() if n.endswith(name)]
            if not matches:
                raise FileNotFoundError(f"{name} was not found in the downloaded archive.")
            (DATA_DIR / name).write_bytes(archive.read(matches[0]))
    print("Dataset downloaded into data/raw/")
    print("Using student-mat.csv (Mathematics course) for this project.")
except Exception as error:
    print("\nAutomatic download failed:", error)
    print("Please download the Student Performance ZIP from:")
    print("https://archive.ics.uci.edu/dataset/320/student%2Bperformance")
    print("Extract student-mat.csv into the data/raw/ folder, then run this script again.")
    raise
