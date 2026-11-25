"""Load and prepare the UCI student dataset."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/student-mat.csv")

FEATURES = [
    "age", "studytime", "failures", "absences", "G1", "G2",
    "schoolsup", "famsup", "paid", "activities", "internet",
    "health", "goout", "freetime"
]

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset missing. First run: python download_data.py"
        )
    # UCI files use semicolon as the separator.
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop_duplicates().copy()

    # Convert final marks into the three categories required by the project.
    # G3 is measured from 0 to 20 in the original dataset.
    df["performance"] = pd.cut(
        df["G3"],
        bins=[-1, 9, 13, 20],
        labels=["Needs Improvement", "Average Performance", "Good Performance"]
    )
    df = df.dropna(subset=["performance"])
    X = df[FEATURES].copy()
    y = df["performance"].astype(str)
    return df, X, y

if __name__ == "__main__":
    data, X, y = load_data()
    print("Rows:", len(data))
    print("Duplicate records removed.")
    print("Missing values by column:\n", X.isnull().sum())
    print("\nTarget categories:\n", y.value_counts())
    print("\nFeature preview:\n", X.head())
