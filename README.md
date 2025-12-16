# Car Price Prediction - Machine Learning Competition

🏆 **3rd Place** out of 34 participants in a private Kaggle-style competition

This project was completed as part of the **Business Analytics & Data Science** course at the **University of St. Gallen** (Bachelor in Economics program), where students competed in a private Kaggle competition to predict car prices using machine learning techniques.

## 🎯 Project Overview

The goal of this competition was to build a machine learning model that accurately predicts car prices based on various features such as brand, model, year, mileage, engine size, fuel type, and more. The evaluation metric was **Root Mean Squared Error (RMSE)** on a hidden test set.

## 🏅 Competition Results

- **Final Ranking**: 3rd place out of 34 participants
- **Best Model RMSE**: 1,892 (LightGBM)
- **Improvement over Baseline**: 53.9% reduction in RMSE (from 4,103 to 1,892)

### Model Performance Summary

| Model | RMSE | Improvement % | Rank |
|-------|------|---------------|------|
| **LightGBM** | **1,892** | 53.9% | 🥇 1 |
| **CatBoost** | 1,982 | 51.7% | 🥈 2 |
| **XGBoost** | 2,476 | 39.7% | 🥉 3 |
| Random Forest | 2,907 | 29.2% | 4 |
| Linear Regression | 4,103 | 0.0% | 5 |

## 🔑 Key Achievements

- Achieved **top 3 performance** (top 8.8%) in the competition
- Implemented advanced feature engineering including target encoding, polynomial features, and tax-based interactions
- Optimized hyperparameters for multiple gradient boosting models (LightGBM, CatBoost, XGBoost)
- Built a robust preprocessing pipeline preventing data leakage
- Created comprehensive visualizations comparing model performance

## 📊 Methodology

### 1. Data Preprocessing
- **Target Transformation**: Applied log transformation (`log1p`) to handle price distribution skewness
- **Outlier Handling**: Used IQR-based clipping for extreme values
- **Missing Values**: Handled missing data appropriately for each feature type

### 2. Feature Engineering
- **Derived Features**: 
  - `car_age`: Years since manufacture
  - `mileage_per_year`: Annual mileage estimate
  - `engine_efficiency`: Engine size to MPG ratio
  - `power_index`: Engine size × MPG
  - `brand_model`: Combined brand and model identifier
- **Target Encoding**: Mean encoding with smoothing for categorical variables
- **Polynomial Features**: Squared terms and interaction features
- **Tax-based Features**: Tax interactions with engine size, MPG, and mileage

### 3. Model Development
- **Baseline Models**: Linear Regression, Random Forest
- **Gradient Boosting Models**: 
  - **LightGBM**: Best performing model with extensive hyperparameter tuning
  - **CatBoost**: Strong performance with native categorical handling
  - **XGBoost**: Competitive results with target encoding
- **Hyperparameter Optimization**: Used `RandomizedSearchCV` for efficient parameter space exploration

### 4. Model-Specific Preprocessing
- **XGBoost**: Target encoding + one-hot encoding for categorical features
- **CatBoost/LightGBM**: Native categorical feature handling with target encoding

## 📁 Project Structure

```
.
├── data/
│   ├── raw/              # Original train and test datasets
│   └── processed/        # Processed datasets (if any)
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_linear_regression.ipynb      # Baseline Linear Regression
│   ├── 03_random_forest.ipynb          # Random Forest Model
│   ├── 04_xgboost.ipynb                 # XGBoost Model
│   ├── 05_catboost.ipynb                # CatBoost Model
│   ├── 06_lightGBM.ipynb                # LightGBM Model (Best Model)
│   └── 08_model_comparison_visualizations.ipynb  # Performance Visualizations
├── src/
│   ├── load_data.py                     # Data loading utilities
│   └── preprocess/
│       ├── preprocessing_pipeline.py    # Main preprocessing pipeline
│       ├── encoding.py                  # Encoding classes (Target, Frequency, OneHot)
│       ├── feature_engineering.py      # Feature creation functions
│       ├── outlier_handling.py          # Outlier detection and handling
│       └── target_transformation.py     # Target variable transformations
├── ensemble/
│   ├── blending.py                      # Ensemble blending utilities
│   ├── config.py                        # Model configurations
│   └── experiment_blending.ipynb        # Blending experiments
├── models/                              # Saved trained models
├── results/                             # Predictions and visualizations
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Assignment 2nd trial"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Models

1. **Exploratory Data Analysis**: Start with `notebooks/01_EDA.ipynb` to understand the data
2. **Baseline Models**: Run `notebooks/02_linear_regression.ipynb` and `notebooks/03_random_forest.ipynb`
3. **Gradient Boosting Models**: 
   - `notebooks/04_xgboost.ipynb` - XGBoost implementation
   - `notebooks/05_catboost.ipynb` - CatBoost implementation
   - `notebooks/06_lightGBM.ipynb` - LightGBM implementation (best model)
4. **Visualizations**: Run `notebooks/08_model_comparison_visualizations.ipynb` to generate performance charts

### Generating Predictions

Each model notebook includes code to:
- Load and preprocess the test data
- Generate predictions using the trained model
- Save predictions to CSV format for submission

## 🛠️ Key Technologies

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning utilities and preprocessing
- **XGBoost**: Gradient boosting framework
- **CatBoost**: Categorical boosting library
- **LightGBM**: Light gradient boosting machine
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development environment

## 📈 Key Insights

1. **Feature Engineering Impact**: Target encoding and polynomial features significantly improved model performance
2. **Model Selection**: LightGBM outperformed other gradient boosting methods, likely due to its efficient handling of categorical features
3. **Hyperparameter Tuning**: Careful regularization tuning was crucial to prevent overfitting
4. **Preprocessing Pipeline**: A robust, reusable preprocessing pipeline was essential for consistency across models

## 📝 Notes

- All models use log-transformed target variables for training, with predictions transformed back to original scale
- The preprocessing pipeline ensures no data leakage by fitting transformations only on training data
- Model-specific preprocessing strategies were employed to leverage each algorithm's strengths

## 👤 Author

**Leonardo Gonnelli**

Bachelor in Economics, University of St. Gallen

Completed as part of the Business Analytics & Data Science course.

## 📄 License

This project was completed for academic purposes as part of a course assignment.

---

*For questions or feedback, please open an issue or contact the author.*

