import pandas as pd
import numpy as np

class FrequencyEncoder:
    """
    Frequency encoding that fits on training data only to prevent data leakage.
    """
    def __init__(self, column_name):
        self.column_name = column_name
        self.frequency_map = {}
        self.fitted = False
    
    def fit(self, df):
        """Compute frequency mapping from training data only."""
        if self.column_name not in df.columns:
            self.fitted = True
            return self
        
        self.frequency_map = df[self.column_name].value_counts().to_dict()
        self.fitted = True
        return self
    
    def transform(self, df):
        """Apply frequency encoding to data."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()
        if self.column_name not in df.columns:
            return df
        
        # Map frequencies, use 0 for unseen values
        freq_col_name = f"{self.column_name}_freq"
        df[freq_col_name] = df[self.column_name].map(self.frequency_map).fillna(0).astype(int)
        df.drop(columns=[self.column_name], inplace=True)
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class OneHotEncoder:
    """
    One-hot encoding that fits on training data to ensure consistent columns.
    """
    def __init__(self, categorical_cols=None):
        self.categorical_cols = categorical_cols or ['brand', 'transmission', 'fuelType']
        self.dummy_columns = []
        self.fitted = False
    
    def fit(self, df):
        """Determine dummy columns from training data."""
        existing_cols = [col for col in self.categorical_cols if col in df.columns]
        
        if not existing_cols:
            self.fitted = True
            return self
        
        # Create dummy variables to see what columns will be created
        dummy_df = pd.get_dummies(df[existing_cols], drop_first=True)
        self.dummy_columns = dummy_df.columns.tolist()
        self.fitted = True
        return self
    
    def transform(self, df):
        """Apply one-hot encoding, ensuring same columns as training."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()
        existing_cols = [col for col in self.categorical_cols if col in df.columns]
        
        if not existing_cols:
            # Add missing dummy columns with zeros
            for col in self.dummy_columns:
                if col not in df.columns:
                    df[col] = 0
            return df
        
        # Create dummies
        dummy_df = pd.get_dummies(df[existing_cols], drop_first=True)
        
        # Drop original categorical columns
        df = df.drop(columns=existing_cols)
        
        # Ensure all training dummy columns exist
        for col in self.dummy_columns:
            if col not in dummy_df.columns:
                dummy_df[col] = 0
        
        # Only keep columns that were in training
        dummy_df = dummy_df[self.dummy_columns]
        
        # Concatenate
        df = pd.concat([df, dummy_df], axis=1)
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


# Backward compatibility functions (deprecated - use classes instead)
def encode_model_frequency(df: pd.DataFrame, enabled=True) -> pd.DataFrame:
    """
    DEPRECATED: This function computes frequencies on input data, causing data leakage.
    Use FrequencyEncoder class with fit/transform pattern instead.
    """
    if not enabled:
        return df
    encoder = FrequencyEncoder('model')
    return encoder.fit_transform(df)


def encode_one_hot(df: pd.DataFrame, enabled=True) -> pd.DataFrame:
    """
    DEPRECATED: This function may create different columns for train/test.
    Use OneHotEncoder class with fit/transform pattern instead.
    """
    if not enabled:
        return df
    encoder = OneHotEncoder()
    return encoder.fit_transform(df)


def encode_brand_model_frequency(df: pd.DataFrame, enabled=True) -> pd.DataFrame:
    """
    DEPRECATED: This function computes frequencies on input data, causing data leakage.
    Use FrequencyEncoder class with fit/transform pattern instead.
    """
    if not enabled:
        return df
    encoder = FrequencyEncoder('brand_model')
    return encoder.fit_transform(df)