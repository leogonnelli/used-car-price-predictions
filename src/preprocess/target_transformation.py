import pandas as pd
import numpy as np

def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a log-transformed target column 'log_price' to the dataframe.
    Useful for models sensitive to skewed target distributions.
    """
    df = df.copy()
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
    return df

def inverse_log_predictions(log_preds: np.ndarray) -> np.ndarray:
    """
    Converts log-transformed predictions back to original scale.
    """
    return np.expm1(log_preds)