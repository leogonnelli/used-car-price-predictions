

import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_FILE, TEST_FILE

def load_train_data() -> pd.DataFrame:
    return pd.read_csv(f"{RAW_DATA_PATH}/{TRAIN_FILE}")

def load_test_data() -> pd.DataFrame:
    return pd.read_csv(f"{RAW_DATA_PATH}/{TEST_FILE}")

def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(f"{PROCESSED_DATA_PATH}/{filename}.csv", index=False)