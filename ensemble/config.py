"""
Configuration for ensemble models.
Update RMSE scores as models improve.
"""
MODEL_CONFIGS = {
    'lightgbm': {
        'path': 'lightgbm_test_preds.csv',
        'rmse': 1892,  # Standalone: 1892.37 RMSE
        'description': 'LightGBM with target encoding, tax features, polynomial features'
    },
    'catboost': {
        'path': 'catboost_test_preds.csv',
        'rmse': 1982,  # Standalone: 1982.05 RMSE
        'description': 'CatBoost with target encoding, tax features, polynomial features'
    },
    'xgboost': {
        'path': 'xgb_test_preds.csv',
        'rmse': 2476,  # Update after running with new features
        'description': 'XGBoost with target encoding, tax features, polynomial features'
    },
    'random_forest': {
        'path': 'tuned_rf_submission.csv',
        'rmse': 2100,  # Approximate, update if needed
        'description': 'Random Forest (may be weaker, consider excluding)'
    }
}

# Best models to use for blending (exclude weaker models)
BEST_MODELS = ['lightgbm', 'catboost']  # Add 'xgboost' if it improves significantly

