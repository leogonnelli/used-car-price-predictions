"""
Blending utilities for ensemble predictions.
Supports weighted averaging with inverse RMSE weighting.
"""
import pandas as pd
import numpy as np
from pathlib import Path


class WeightedBlender:
    """
    Weighted blending of multiple model predictions.
    """
    
    def __init__(self, model_configs):
        """
        Initialize blender with model configurations.
        
        Parameters:
        -----------
        model_configs : dict
            Dictionary mapping model names to config dicts with:
            - 'path': path to prediction CSV file
            - 'rmse': validation/test RMSE score (for inverse weighting)
            - 'weight': optional manual weight (overrides inverse RMSE)
        """
        self.model_configs = model_configs
        self.weights = None
        self.predictions = {}
        
    def load_predictions(self, results_dir="../results"):
        """
        Load all model predictions from CSV files.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing prediction CSV files.
        """
        results_path = Path(results_dir)
        
        for model_name, config in self.model_configs.items():
            pred_path = results_path / config['path']
            if not pred_path.exists():
                raise FileNotFoundError(f"Prediction file not found: {pred_path}")
            
            df = pd.read_csv(pred_path)
            # Ensure ID column exists and is set as index
            if 'ID' in df.columns:
                df = df.set_index('ID')
            elif df.index.name == 'ID':
                pass
            else:
                raise ValueError(f"Could not find ID column in {pred_path}")
            
            # Get predictions column (should be 'Actual')
            if 'Actual' not in df.columns:
                raise ValueError(f"Could not find 'Actual' column in {pred_path}")
            
            self.predictions[model_name] = df['Actual']
            print(f"✅ Loaded {model_name}: {len(df)} predictions")
        
        # Verify all predictions have same index
        indices = [pred.index for pred in self.predictions.values()]
        if not all(idx.equals(indices[0]) for idx in indices):
            raise ValueError("All predictions must have the same ID indices")
        
        print(f"\n✅ All predictions loaded. Shape: {len(indices[0])} samples")
        return self
    
    def compute_weights(self, method='inverse_rmse'):
        """
        Compute blending weights.
        
        Parameters:
        -----------
        method : str
            'inverse_rmse': weight inversely proportional to RMSE
            'inverse_rmse_squared': weight inversely proportional to RMSE squared (more aggressive)
            'equal': equal weights for all models
            'manual': use weights from config
            'best_only': use only the best model (weight=1.0)
        """
        if method == 'inverse_rmse':
            # Weight inversely proportional to RMSE
            inv_rmse = {name: 1.0 / config['rmse'] 
                       for name, config in self.model_configs.items()}
            total = sum(inv_rmse.values())
            self.weights = {name: w / total for name, w in inv_rmse.items()}
            
        elif method == 'inverse_rmse_squared':
            # Weight inversely proportional to RMSE squared (more aggressive, favors best model)
            inv_rmse_sq = {name: 1.0 / (config['rmse'] ** 2)
                          for name, config in self.model_configs.items()}
            total = sum(inv_rmse_sq.values())
            self.weights = {name: w / total for name, w in inv_rmse_sq.items()}
            
        elif method == 'equal':
            n_models = len(self.model_configs)
            self.weights = {name: 1.0 / n_models 
                          for name in self.model_configs.keys()}
            
        elif method == 'manual':
            # Use manual weights from config
            self.weights = {name: config.get('weight', 1.0) 
                          for name, config in self.model_configs.items()}
            total = sum(self.weights.values())
            self.weights = {name: w / total for name, w in self.weights.items()}
            
        elif method == 'best_only':
            # Use only the best model (lowest RMSE)
            best_model = min(self.model_configs.items(), key=lambda x: x[1]['rmse'])[0]
            self.weights = {name: 1.0 if name == best_model else 0.0 
                          for name in self.model_configs.keys()}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print("\n📊 Blending weights:")
        for name, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            rmse = self.model_configs[name]['rmse']
            print(f"  {name:15s}: {weight:.4f} (RMSE: {rmse:.2f})")
        
        return self
    
    def blend(self):
        """
        Blend predictions using computed weights.
        
        Returns:
        --------
        pd.Series
            Blended predictions indexed by ID.
        """
        if self.weights is None:
            raise ValueError("Must compute weights first. Call compute_weights()")
        
        if not self.predictions:
            raise ValueError("Must load predictions first. Call load_predictions()")
        
        # Weighted average
        blended = pd.Series(0.0, index=list(self.predictions.values())[0].index)
        for model_name, preds in self.predictions.items():
            blended += self.weights[model_name] * preds
        
        print(f"\n✅ Blended predictions: min={blended.min():.2f}, max={blended.max():.2f}, mean={blended.mean():.2f}")
        return blended
    
    def save_blended(self, output_path, test_ids=None):
        """
        Save blended predictions to CSV in Kaggle submission format.
        
        Parameters:
        -----------
        output_path : str
            Path to save submission CSV.
        test_ids : pd.Series, optional
            If provided, use these IDs (useful if predictions were filtered).
        """
        blended = self.blend()
        
        if test_ids is not None:
            # Ensure we have predictions for all test IDs
            submission = pd.DataFrame({
                'ID': test_ids,
                'Actual': blended.reindex(test_ids).values
            })
        else:
            submission = pd.DataFrame({
                'ID': blended.index,
                'Actual': blended.values
            })
        
        submission.to_csv(output_path, index=False)
        print(f"✅ Saved blended predictions to: {output_path}")
        return submission


def create_submission(predictions, output_path, test_ids=None):
    """
    Helper function to create Kaggle submission file.
    
    Parameters:
    -----------
    predictions : pd.Series or np.array
        Predictions (should be in original price scale, not log).
    output_path : str
        Path to save submission CSV.
    test_ids : pd.Series, optional
        Test IDs if predictions don't have index.
    """
    if isinstance(predictions, pd.Series):
        submission = pd.DataFrame({
            'ID': predictions.index,
            'Actual': predictions.values
        })
    else:
        if test_ids is None:
            raise ValueError("test_ids required when predictions is not a Series")
        submission = pd.DataFrame({
            'ID': test_ids,
            'Actual': predictions
        })
    
    submission.to_csv(output_path, index=False)
    print(f"✅ Saved submission to: {output_path}")
    return submission

