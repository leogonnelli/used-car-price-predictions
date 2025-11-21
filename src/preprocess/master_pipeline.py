"""
Master preprocessing pipeline with backward compatibility.
For new code, use PreprocessingPipeline class to prevent data leakage.
"""
from src.preprocess.preprocessing_pipeline import PreprocessingPipeline

# Global pipeline instance for backward compatibility (WARNING: causes data leakage)
_backward_compat_pipeline = None


def preprocess(df, drop_year=True, use_log_target=False, drop_low_importance=False, encode_data=True):
    """
    DEPRECATED: This function causes data leakage by computing statistics on input data.
    
    For proper preprocessing without data leakage, use PreprocessingPipeline class:
    
    Example:
        pipeline = PreprocessingPipeline(use_log_target=True, encode_data=True)
        train_processed = pipeline.fit_transform(train_df)
        test_processed = pipeline.transform(test_df)
    
    This function is kept for backward compatibility but should not be used for production.
    It computes statistics (quantiles, frequencies) on the input dataframe, which causes
    data leakage when preprocessing train and test data separately.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw input dataframe.
    drop_year : bool
        If True, drops the 'year' column after creating 'car_age' (deprecated, always True).
    use_log_target : bool
        If True, adds 'log_price' as a transformed target.
    drop_low_importance : bool
        If True, drops low-importance features.
    encode_data : bool
        If True, applies encoding steps to categorical features; 
        if False, skips encoding (useful for models like CatBoost).

    Returns:
    --------
    pd.DataFrame
        Fully preprocessed dataframe.
    """
    # Use the new pipeline for backward compatibility
    # NOTE: This still causes leakage if called separately on train/test
    pipeline = PreprocessingPipeline(
        use_log_target=use_log_target,
        drop_low_importance=drop_low_importance,
        encode_data=encode_data
    )
    return pipeline.fit_transform(df)
