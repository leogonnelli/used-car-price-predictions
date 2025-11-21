"""
PreprocessingPipeline class that prevents data leakage by fitting on training data
and transforming test data separately.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from src.preprocess.clean_columns import drop_useless_columns
from src.preprocess.outlier_handling import OutlierClipper
from src.preprocess.encoding import FrequencyEncoder, OneHotEncoder
from src.preprocess.feature_engineering import (
    add_car_age,
    add_mileage_per_year,
    add_engine_efficiency,
    add_power_index,
    add_age_mileage_interaction,
    add_log_mileage,
    add_brand_model
)
from src.preprocess.target_transformation import add_log_target


class PreprocessingPipeline:
    """
    A preprocessing pipeline that fits on training data and transforms test data.
    Prevents data leakage by computing statistics only on training data.
    """
    
    def __init__(self, 
                 use_log_target=False, 
                 drop_low_importance=False, 
                 encode_data=True,
                 current_year=None):
        """
        Initialize the preprocessing pipeline.
        
        Parameters:
        -----------
        use_log_target : bool
            If True, adds 'log_price' as a transformed target.
        drop_low_importance : bool
            If True, drops low-importance features.
        encode_data : bool
            If True, applies encoding steps to categorical features.
        current_year : int, optional
            Year to use for computing car_age. If None, uses current year.
        """
        self.use_log_target = use_log_target
        self.drop_low_importance = drop_low_importance
        self.encode_data = encode_data
        self.current_year = current_year or datetime.now().year
        
        # Components that need fitting
        self.outlier_clipper = None
        self.model_freq_encoder = None
        self.brand_model_freq_encoder = None
        self.one_hot_encoder = None
        
        self.fitted = False
        self.low_importance_features = [
            "transmission_Semi-Auto",
            "brand_VW",
            "model_freq",
            "brand_model_freq"
        ]
    
    def _apply_feature_engineering(self, df):
        """Apply feature engineering steps (no fitting needed)."""
        df = df.copy()
        df = add_car_age(df, current_year=self.current_year)
        df = add_mileage_per_year(df)
        df = add_engine_efficiency(df)
        df = add_power_index(df)
        df = add_age_mileage_interaction(df)
        df = add_log_mileage(df)
        return df
    
    def fit(self, df):
        """
        Fit the pipeline on training data.
        Computes all statistics and mappings from training data only.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe with 'price' column.
        
        Returns:
        --------
        self : PreprocessingPipeline
        """
        df = df.copy()
        
        # Step 1: Drop useless columns
        df = drop_useless_columns(df)
        
        # Step 2: Apply feature engineering (no fitting needed)
        df = self._apply_feature_engineering(df)
        
        # Step 3: Fit outlier clipper on training data
        self.outlier_clipper = OutlierClipper()
        df = self.outlier_clipper.fit_transform(df)
        
        # Step 4: Handle encoding if needed
        if self.encode_data:
            # Add brand_model feature
            df = add_brand_model(df)
            
            # Fit frequency encoders on training data
            if 'brand_model' in df.columns:
                self.brand_model_freq_encoder = FrequencyEncoder('brand_model')
                df = self.brand_model_freq_encoder.fit_transform(df)
            
            if 'model' in df.columns:
                self.model_freq_encoder = FrequencyEncoder('model')
                df = self.model_freq_encoder.fit_transform(df)
            
            # Fit one-hot encoder on training data
            self.one_hot_encoder = OneHotEncoder()
            df = self.one_hot_encoder.fit_transform(df)
        else:
            # Ensure categorical columns are strings
            for col in ["brand", "model", "transmission", "fuelType"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        # Step 5: Add log target if needed
        if self.use_log_target:
            df = add_log_target(df)
        
        # Step 6: Drop low importance features if needed
        if self.drop_low_importance:
            df.drop(columns=[col for col in self.low_importance_features 
                            if col in df.columns], inplace=True)
        
        # Step 7: Drop rows with missing values
        df.dropna(inplace=True)
        
        # Store the transformed data for fit_transform()
        self._fitted_data = df
        
        self.fitted = True
        return self
    
    def transform(self, df):
        """
        Transform data using fitted parameters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to transform (test data).
        
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()
        
        # Step 1: Drop useless columns
        df = drop_useless_columns(df)
        
        # Step 2: Apply feature engineering
        df = self._apply_feature_engineering(df)
        
        # Step 3: Apply outlier clipping (using training bounds)
        df = self.outlier_clipper.transform(df)
        
        # Step 4: Handle encoding if needed
        if self.encode_data:
            # Add brand_model feature
            df = add_brand_model(df)
            
            # Apply frequency encoders
            if self.brand_model_freq_encoder is not None:
                df = self.brand_model_freq_encoder.transform(df)
            
            if self.model_freq_encoder is not None:
                df = self.model_freq_encoder.transform(df)
            
            # Apply one-hot encoder
            if self.one_hot_encoder is not None:
                df = self.one_hot_encoder.transform(df)
        else:
            # Ensure categorical columns are strings
            for col in ["brand", "model", "transmission", "fuelType"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        # Step 5: Add log target if needed (for consistency, though test won't have price)
        if self.use_log_target and 'price' in df.columns:
            df = add_log_target(df)
        
        # Step 6: Drop low importance features if needed
        if self.drop_low_importance:
            df.drop(columns=[col for col in self.low_importance_features 
                            if col in df.columns], inplace=True)
        
        # Step 7: Drop rows with missing values
        df.dropna(inplace=True)
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step (for training data)."""
        self.fit(df)
        return self._fitted_data.copy()

