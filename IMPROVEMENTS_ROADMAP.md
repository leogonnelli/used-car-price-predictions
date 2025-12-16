# Improvement Roadmap: 1889 → 1750 RMSE

Current best: **1889 RMSE** (90% LightGBM + 10% CatBoost)  
Target: **1750 RMSE** (competition best)  
Gap: **~140 RMSE to gain**

## 🎯 High-Priority Improvements (Expected: 30-60 RMSE gain)

### 1. Advanced Target Encoding Combinations ⭐⭐⭐
**Expected gain: 20-40 RMSE**

Add target encoding for categorical combinations:
- `(brand, transmission)` → `brand_transmission_target_enc`
- `(brand, fuelType)` → `brand_fuelType_target_enc`
- `(model, transmission)` → `model_transmission_target_enc`
- `(fuelType, transmission)` → `fuelType_transmission_target_enc`

**Why:** Captures interactions between categoricals that single encodings miss.

**Implementation:**
- Extend `TargetEncoder` to handle multiple columns
- Add to `PreprocessingPipeline` when `use_target_encoding=True`

### 2. Statistical Aggregation Features ⭐⭐⭐
**Expected gain: 15-30 RMSE**

Add group-based statistics:
- Mean/std price per brand
- Mean/std price per model
- Count per brand/model (frequency)
- Price percentile per brand/model

**Why:** Provides context about category-level price distributions.

**Implementation:**
- Create `StatisticalAggregator` class
- Add to feature engineering pipeline

### 3. More Interaction Features ⭐⭐
**Expected gain: 10-20 RMSE**

Add missing interactions:
- `tax * car_age`
- `tax * mileage`
- `engineSize * mileage`
- `mpg * mileage`
- `tax * mpg`
- Three-way: `engineSize * mpg * car_age`

**Why:** Captures non-linear relationships between key features.

**Implementation:**
- Extend `add_polynomial_features()` function

## 🔧 Medium-Priority Improvements (Expected: 20-40 RMSE gain)

### 4. Cross-Validation for Blend Weights ⭐⭐
**Expected gain: 5-15 RMSE**

Use K-fold CV to find optimal blend ratios instead of manual tuning.

**Why:** More systematic approach to finding best weights.

**Implementation:**
- Create `CVBlendingOptimizer` class
- Use 5-fold CV to test different weight combinations
- Find weights that minimize CV RMSE

### 5. Stacking with Meta-Learner ⭐⭐
**Expected gain: 10-20 RMSE**

Replace simple blending with stacking:
- Train LightGBM, CatBoost, XGBoost on training set
- Generate OOF predictions
- Train meta-learner (Ridge/ElasticNet) on OOF predictions
- Meta-learner learns optimal combination

**Why:** More sophisticated than weighted averaging.

**Implementation:**
- Create `StackingEnsemble` class
- Use out-of-fold predictions to prevent overfitting

### 6. LightGBM Hyperparameter Fine-Tuning ⭐⭐
**Expected gain: 5-15 RMSE**

Expand hyperparameter search around current best config:
- Test more `num_leaves` values (current: ~31)
- Test more `learning_rate` values (current: ~0.05)
- Test `feature_fraction`, `bagging_fraction`
- Test different `min_data_in_leaf`

**Why:** Current search might not have found global optimum.

## 📊 Lower-Priority Improvements (Expected: 10-30 RMSE gain)

### 7. Feature Selection Based on Importance ⭐
**Expected gain: 5-10 RMSE**

Remove low-importance features:
- Get feature importance from LightGBM
- Remove features with importance < threshold
- Retrain with reduced feature set

**Why:** Reduces noise and overfitting.

### 8. Advanced Outlier Handling ⭐
**Expected gain: 3-8 RMSE**

Improve outlier detection:
- Use IQR-based detection per feature
- Use isolation forest for multivariate outliers
- Cap outliers instead of clipping

**Why:** Better outlier handling can improve model performance.

### 9. Pseudo-Labeling (Advanced) ⭐
**Expected gain: 5-15 RMSE** (risky, can hurt if done wrong)

Use confident test predictions to augment training:
- Train initial model
- Predict on test set
- Add high-confidence predictions to training
- Retrain

**Why:** Effectively increases training data size.

**Warning:** Can cause overfitting if not done carefully.

## 🚀 Quick Wins (Do First)

1. **Add more interaction features** (30 min)
   - Extend `add_polynomial_features()` with tax interactions
   
2. **Add statistical aggregation** (1-2 hours)
   - Create `StatisticalAggregator` class
   - Add mean/std per brand/model

3. **Advanced target encoding** (2-3 hours)
   - Extend `TargetEncoder` for combinations
   - Add to pipeline

## 📈 Expected Total Gain

If all high-priority improvements are implemented:
- **Conservative estimate:** 50-80 RMSE improvement → **~1810-1840 RMSE**
- **Optimistic estimate:** 80-120 RMSE improvement → **~1770-1810 RMSE**

This would get you very close to the 1750 target!

## 🎯 Recommended Order

1. **Week 1:** Quick wins (interactions, statistical features)
2. **Week 2:** Advanced target encoding
3. **Week 3:** CV blending + Stacking
4. **Week 4:** Fine-tuning + feature selection

## 💡 Pro Tips

- **Test incrementally:** Add one improvement at a time, measure impact
- **Use XGBoost as proxy:** Test features on XGBoost (faster) before LightGBM
- **Monitor overfitting:** Watch validation vs test RMSE gap
- **Keep it simple:** Don't overcomplicate - sometimes simpler is better


