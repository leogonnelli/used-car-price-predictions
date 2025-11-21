import pandas as pd
import numpy as np

class OutlierClipper:
    """
    Clips outliers based on quantiles computed on training data only.
    Prevents data leakage by fitting on training data and transforming test data.
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.clip_bounds = {}
        self.fitted = False
    
    def fit(self, df):
        """Compute clipping bounds from training data only."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            if col == 'price' or col == 'log_price':
                continue
            self.clip_bounds[col] = {
                'lower': df[col].quantile(self.lower_quantile),
                'upper': df[col].quantile(self.upper_quantile)
            }
        
        self.fitted = True
        return self
    
    def transform(self, df):
        """Apply clipping bounds to data."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()
        for col, bounds in self.clip_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(bounds['lower'], bounds['upper'])
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


# Backward compatibility function (deprecated - use OutlierClipper class)
def clip_outliers(df, lower_quantile=0.01, upper_quantile=0.99):
    """
    DEPRECATED: This function computes quantiles on the input data, causing data leakage.
    Use OutlierClipper class with fit/transform pattern instead.
    """
    clipper = OutlierClipper(lower_quantile, upper_quantile)
    return clipper.fit_transform(df)
