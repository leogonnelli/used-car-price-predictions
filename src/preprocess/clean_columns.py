import pandas as pd

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops non-informative or irrelevant columns from the dataset.
    Currently drops:
    - 'ID' column if present

    Parameters:
    df (pd.DataFrame): Raw input dataframe

    Returns:
    pd.DataFrame: Cleaned dataframe with dropped columns
    """
    df = df.copy()

    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    return df
