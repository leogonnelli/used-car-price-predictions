# Ensemble Blending

This folder contains utilities and experiments for blending multiple model predictions to improve performance.

## Current Best Models

- **LightGBM**: 1892 RMSE (3rd place! 🎉)
- **CatBoost**: ~2023 RMSE (update after running with new features)
- **XGBoost**: ~2476 RMSE (update after running with new features)

## Files

- `blending.py`: Core blending utilities (`WeightedBlender` class)
- `config.py`: Model configurations (RMSE scores, file paths)
- `experiment_blending.ipynb`: Jupyter notebook for experimentation
- `README.md`: This file

## Quick Start

1. **Update RMSE scores** in `config.py` after running models with new features
2. **Open** `experiment_blending.ipynb`
3. **Run cells** to experiment with different blending methods
4. **Save best blend** to `../results/ensemble_blended_inv_rmse.csv`

## Blending Methods

### 1. Inverse RMSE Weighting
Weights models inversely proportional to their RMSE. Better models get higher weights.

```python
weight_i = (1 / RMSE_i) / sum(1 / RMSE_j for all j)
```

### 2. Inverse RMSE Squared (More Aggressive)
Squares the inverse RMSE to give even more weight to better models.

```python
weight_i = (1 / RMSE_i²) / sum(1 / RMSE_j² for all j)
```

### 3. Equal Weighting
Simple average of all models.

### 4. Manual Weighting (Recommended for High Correlation)
Experiment with custom weights. For highly correlated models, use 90-95% for the best model.

### 5. Best Only
Use only the best model (useful for comparison).

## Expected Improvement

**Important Finding:** Models are highly correlated (0.997), which limits blending benefits.

**Current Results:**
- LightGBM alone: **1892.37 RMSE**
- CatBoost: 1982.05 RMSE
- 50/50 blend: 1902.41 RMSE (worse than LightGBM alone!)
- **90% LightGBM blend: 1889.03 RMSE** 🎉 (BEST - 3.3 RMSE improvement!)
- 95% LightGBM blend: 1890.35 RMSE (slightly worse than 90%)

**Best Blend Found:**
- **90% LightGBM + 10% CatBoost = 1889.03 RMSE**
- This captures small complementary signals from CatBoost without being dragged down

**Recommendation:**
- Fine-tune around 90% (test 87-92%) to find the exact optimal ratio
- The 90% blend is currently the best submission

## Usage Example

```python
from ensemble.blending import WeightedBlender
from ensemble.config import MODEL_CONFIGS, BEST_MODELS

# Use best models only
best_configs = {name: MODEL_CONFIGS[name] for name in BEST_MODELS}

# Create blender and load predictions
blender = WeightedBlender(best_configs)
blender.load_predictions(results_dir="../results")

# Compute weights and blend
blender.compute_weights(method='inverse_rmse')
blended = blender.blend()

# Save submission
blender.save_blended(output_path="../results/ensemble_blended.csv")
```

## Notes

- All predictions should be in **original price scale** (not log-transformed)
- Ensure all prediction files have `ID` and `Actual` columns
- Update RMSE scores in `config.py` as models improve
- Exclude weaker models (e.g., Random Forest) if they hurt performance

