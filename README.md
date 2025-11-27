# 🏠 Immo-Eliza ML: Real Estate Price Prediction

A machine learning project that predicts real estate prices in Belgium using the Immovlan dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)

---

## 📋 Table of Contents

- [Description](#description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 📖 Description

This project builds a machine learning model to predict property prices in Belgium. The model is trained on the Immovlan dataset containing real estate listings with features such as number of bedrooms, property type, region, and amenities.

### Learning Objectives

- ✅ Preprocess data for machine learning
- ✅ Apply linear regression in a real-life context
- ✅ Explore multiple machine learning models for regression
- ✅ Evaluate model performance using appropriate metrics

---

## 📁 Project Structure

```
immo-eliza-ML/
├── Model/                      # Saved model and preprocessing objects
│   ├── model.joblib            # Trained XGBoost model
│   ├── scaler.joblib           # StandardScaler for numeric features
│   ├── target_encoder.joblib   # TargetEncoder for categorical features
│   ├── numeric_imputer.joblib  # SimpleImputer for numeric columns
│   ├── knn_imputer.joblib      # KNNImputer for binary columns
│   ├── feature_order.joblib    # Column order for prediction
│   └── metrics.joblib          # Validation metrics
├── immovlan_cleaned_file.csv   # Cleaned dataset
├── ml_model_Final.ipynb        # Jupyter notebook with EDA and experiments
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/immo-eliza-ML.git
   cd immo-eliza-ML
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Training the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train the XGBoost model
- Save all preprocessing objects and the model to the `Model/` folder
- Display validation metrics

### Making Predictions

```bash
python predict.py
```

Example output:
```
==================================================
PREDICTION RESULT
==================================================

💰 Predicted Price: €350,000.00

📊 Model Performance:
   R²:   0.6452
   MAE:  ±€109,451
   RMSE: ±€276,602

📍 Price Range (using MAE):
   Low:  €240,549.00
   High: €459,451.00
==================================================
```

---

## 🔧 Data Preprocessing

The preprocessing pipeline includes the following steps:

| Step | Method | Description |
|------|--------|-------------|
| 1. Split | 60/20/20 | Train, validation, test split |
| 2. Categorical Encoding | Target Encoding | Encode `type` and `Region` with mean prices |
| 3. Numeric Imputation | Median | Fill missing values in bedrooms, bathrooms, toilets |
| 4. Scaling | StandardScaler | Standardize numeric features (mean=0, std=1) |
| 5. Binary Imputation (KNN) | KNNImputer | Fill `Elevator` and `Garden` using neighbors |
| 6. Binary Imputation (Zero) | fillna(0) | Fill `Garage` and `Swimming pool` with 0 |

### Features Used

| Feature | Type | Preprocessing |
|---------|------|---------------|
| Number of bedrooms | Numeric | Median imputation, Scaling |
| Number of bathrooms | Numeric | Median imputation, Scaling |
| Number of toilets | Numeric | Median imputation, Scaling |
| type | Categorical | Target encoding |
| Region | Categorical | Target encoding |
| Elevator | Binary | KNN imputation |
| Garden | Binary | KNN imputation |
| Garage | Binary | Zero imputation |
| Swimming pool | Binary | Zero imputation |
| postal_code | Numeric | Passed through |

---

## 🤖 Models Evaluated

Four regression models were tested and compared:

| Model | Train R² | Validation R² | Overfitting? |
|-------|----------|---------------|--------------|
| Linear Regression | ~0.33 | ~0.24 | ✓ No |
| Decision Tree | ~0.89 | ~0.52 | ⚠️ Yes |
| Random Forest | ~0.92 | ~0.62 | ⚠️ Yes |
| **XGBoost** | ~0.75 | **~0.65** | ✓ Mild |

**Selected Model: XGBoost** — Best balance between performance and generalization.

### XGBoost Hyperparameters

```python
XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## 📊 Results

### Final Model Performance (Validation Set)

| Metric | Value |
|--------|-------|
| **R²** | 0.6452 |
| **RMSE** | €276,602 |
| **MAE** | €109,451 |

### Interpretation

- **R² = 0.65**: The model explains about 65% of the variance in property prices
- **MAE = €109,451**: On average, predictions are off by about €109,000
- **RMSE = €276,602**: Larger errors are penalized more heavily

### Feature Importance

The most important features for predicting price (from XGBoost):

1. `type` — Property type has the strongest impact
2. `Region` — Location significantly affects price
3. `Number of bedrooms` — More bedrooms = higher price
4. `Swimming pool` — Premium amenity
5. `Garden` — Outdoor space adds value

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- [ ] Cross-validation for more robust evaluation
- [ ] Add more features (e.g., Livable surface, Number of facades)
- [ ] Final test deployment
- [ ] Deploy model as a web API

---

## 👤 Author

**Astha**

---

## 📝 License

This project is part of the BeCode AI Bootcamp.
