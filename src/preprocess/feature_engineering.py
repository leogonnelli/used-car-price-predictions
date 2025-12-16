import pandas as pd
import numpy as np

def add_car_age(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """
    Adds a 'car_age' column calculated from the 'year' column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'year' column
    current_year : int, optional
        Year to use for calculating age. If None, uses 2025 (for backward compatibility).
    """
    df = df.copy()
    if 'year' in df.columns:
        if current_year is None:
            current_year = 2025  # Backward compatibility
        df['car_age'] = current_year - df['year']
        df.drop(columns=['year'], inplace=True)
    return df

def add_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'mileage_per_year' feature: mileage divided by car age (+1 to avoid division by zero).
    """
    df = df.copy()
    if 'mileage' in df.columns and 'car_age' in df.columns:
        df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
    return df

def add_engine_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'engine_efficiency' feature: mpg divided by engine size (+0.1 to avoid division by zero).
    """
    df = df.copy()
    if 'mpg' in df.columns and 'engineSize' in df.columns:
        df['engine_efficiency'] = df['mpg'] / (df['engineSize'] + 0.1)
    return df


# New feature engineering functions
def add_power_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'power_index' feature: engine size multiplied by mpg.
    """
    df = df.copy()
    if 'engineSize' in df.columns and 'mpg' in df.columns:
        df['power_index'] = df['engineSize'] * df['mpg']
    return df

def add_age_mileage_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'age_mileage_interaction' feature: car age multiplied by mileage.
    """
    df = df.copy()
    if 'car_age' in df.columns and 'mileage' in df.columns:
        df['age_mileage_interaction'] = df['car_age'] * df['mileage']
    return df

def add_log_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'log_mileage' feature: natural log of (1 + mileage).
    """
    df = df.copy()
    if 'mileage' in df.columns:
        df['log_mileage'] = df['mileage'].apply(lambda x: pd.NA if x < 0 else np.log1p(x))
    return df

def add_brand_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'brand_model' feature by combining 'brand' and 'model' columns.
    """
    df = df.copy()
    if 'brand' in df.columns and 'model' in df.columns:
        df['brand_model'] = df['brand'].astype(str) + "_" + df['model'].astype(str)
    return df


def add_tax_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds tax-based features that correlate with price segments.
    Tax often relates to emissions/engine size and can indicate market value.
    """
    df = df.copy()
    if 'tax' not in df.columns:
        return df
    
    # Tax efficiency features
    if 'engineSize' in df.columns:
        df['tax_per_engine'] = df['tax'] / (df['engineSize'] + 0.1)
    
    if 'mpg' in df.columns:
        df['tax_per_mpg'] = df['tax'] / (df['mpg'] + 0.1)
    
    if 'car_age' in df.columns:
        df['tax_per_year'] = df['tax'] / (df['car_age'] + 1)
    
    # Tax interactions
    if 'log_mileage' in df.columns:
        df['tax_mileage'] = df['tax'] * df['log_mileage']
    elif 'mileage' in df.columns:
        df['tax_mileage'] = df['tax'] * np.log1p(df['mileage'])
    
    if 'engineSize' in df.columns:
        df['tax_engine'] = df['tax'] * df['engineSize']
    
    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds polynomial and interaction features for key numeric columns.
    Helps capture non-linear relationships.
    """
    df = df.copy()
    
    # Squared features for key variables
    if 'engineSize' in df.columns:
        df['engineSize_squared'] = df['engineSize'] ** 2
    
    if 'mpg' in df.columns:
        df['mpg_squared'] = df['mpg'] ** 2
    
    if 'mileage' in df.columns:
        df['mileage_squared'] = df['mileage'] ** 2
    
    if 'car_age' in df.columns:
        df['car_age_squared'] = df['car_age'] ** 2
    
    # Key interactions
    if 'engineSize' in df.columns and 'mpg' in df.columns:
        df['engine_mpg_interaction'] = df['engineSize'] * df['mpg']
    
    if 'car_age' in df.columns and 'engineSize' in df.columns:
        df['age_engine_interaction'] = df['car_age'] * df['engineSize']
    
    if 'car_age' in df.columns and 'mpg' in df.columns:
        df['age_mpg_interaction'] = df['car_age'] * df['mpg']
    
    return df