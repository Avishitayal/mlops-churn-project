from pathlib import Path

import pandas as pd


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "churn.csv"


def load_data():
    """Load churn dataset from CSV."""
    
    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())