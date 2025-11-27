
# ============================================
# SECTION 1: IMPORTS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# ============================================
# SECTION 2: PREPROCESSING FUNCTIONS
# ============================================

# 1. Split data - into train/val/test (60/20/20)

def split_60_20_20(df, target, random_state=42):

    X = df.drop(target, axis=1)
    y = df[target]

    # Step 1: Train+Val vs Test (80% / 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    # Step 2: Train vs Val (75% / 25% of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# 2. Target encoding for categorical columns (type, Region)

def target_encode(X_train, X_val, X_test, y_train, categorical_cols):
    if not categorical_cols:
        return X_train, X_val, X_test, None
    
    enc = TargetEncoder(smooth="auto", target_type="continuous")
    
    X_train[categorical_cols] = enc.fit_transform(X_train[categorical_cols], y_train)
    X_val[categorical_cols] = enc.transform(X_val[categorical_cols])
    X_test[categorical_cols] = enc.transform(X_test[categorical_cols])
    
    return X_train, X_val, X_test, enc

# 3. Median imputation for numeric columns

def impute_numeric(X_train, X_val, X_test, numeric_cols):
    if not numeric_cols:
        return X_train, X_val, X_test, None
    
    imputer = SimpleImputer(strategy='median')

    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    return X_train, X_val, X_test, imputer

# 4. Outlier handling based on skewness

def analyze_skewness(df, cols): 
    skew_report = {}
    for col in cols:
        s = df[col].skew()
        if abs(s) > 1:
            skew_report[col] = "log_sigma"
        elif abs(s) > 0.5:
            skew_report[col] = "quantile"
        else:
            skew_report[col] = "zscore"
    return skew_report


def log_sigma_cap(s):
    log_s = np.log1p(s)
    mean, std = log_s.mean(), log_s.std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    return np.expm1(np.clip(log_s, lower, upper))


def quantile_cap(s, low_q=0.01, high_q=0.99):
    return s.clip(s.quantile(low_q), s.quantile(high_q))


def zscore_cap(s, z=3):
    mean, std = s.mean(), s.std()
    lower = mean - z * std
    upper = mean + z * std
    return s.clip(lower, upper)


def handle_outliers(X_train, X_val, X_test, numeric_cols):
    if not numeric_cols:
        return X_train, X_val, X_test
    
    skewness = analyze_skewness(X_train, numeric_cols)

    for col, method in skewness.items():
        if method == "log_sigma":
            X_train[col] = log_sigma_cap(X_train[col])
            X_val[col] = log_sigma_cap(X_val[col])
            X_test[col] = log_sigma_cap(X_test[col])
        elif method == "quantile":
            X_train[col] = quantile_cap(X_train[col])
            X_val[col] = quantile_cap(X_val[col])
            X_test[col] = quantile_cap(X_test[col])
        else:  # zscore
            X_train[col] = zscore_cap(X_train[col])
            X_val[col] = zscore_cap(X_val[col])
            X_test[col] = zscore_cap(X_test[col])

    return X_train, X_val, X_test

# 5. Standardization for numeric columns

def scale_numeric(X_train, X_val, X_test, numeric_cols):
    if not numeric_cols:
        return X_train, X_val, X_test, None
    
    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_val, X_test, scaler

# 6. KNN imputation for binary columns with moderate missing (Elevator, Garden)

def knn_impute_binary(X_train, X_val, X_test, binary_cols, n_neighbors=5):
    if not binary_cols:
        return X_train, X_val, X_test, None
    
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    X_train[binary_cols] = knn_imputer.fit_transform(X_train[binary_cols])
    X_val[binary_cols] = knn_imputer.transform(X_val[binary_cols])
    X_test[binary_cols] = knn_imputer.transform(X_test[binary_cols])

    # Round to ensure strictly 0/1 values
    for col in binary_cols:
        X_train[col] = X_train[col].round()
        X_val[col] = X_val[col].round()
        X_test[col] = X_test[col].round()

    return X_train, X_val, X_test, knn_imputer

# 7. Zero imputation for binary columns with high missing (Garage, Swimming pool)

def impute_zero(X_train, X_val, X_test, binary_cols):
    if not binary_cols:
        return X_train, X_val, X_test
    
    for col in binary_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0)
            X_val[col] = X_val[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)

    return X_train, X_val, X_test

# 8. Preprocess Pipeline 

def preprocess(df, target, numeric_cols, categorical_cols=[], binary_cols_knn=[], binary_cols_zero=[]):
    """
    Full preprocessing pipeline.
    Returns all fitted objects needed for predictions.
    """
    
    # 1. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(df, target, random_state=42)
    
    # 2. Target encoding
    target_encoder = None
    if categorical_cols:
        X_train, X_val, X_test, target_encoder = target_encode(X_train, X_val, X_test, y_train, categorical_cols)
    
    # 3. Numeric imputation
    X_train, X_val, X_test, numeric_imputer = impute_numeric(X_train, X_val, X_test, numeric_cols)
    
    # 4. Outlier handling
    X_train, X_val, X_test = handle_outliers(X_train, X_val, X_test, numeric_cols)
    
    # 5. Scaling
    X_train, X_val, X_test, scaler = scale_numeric(X_train, X_val, X_test, numeric_cols)
    
    # 6. KNN imputation for binary columns
    knn_imputer = None
    if binary_cols_knn:
        X_train, X_val, X_test, knn_imputer = knn_impute_binary(X_train, X_val, X_test, binary_cols_knn)
    
    # 7. Zero imputation for binary columns
    if binary_cols_zero:
        X_train, X_val, X_test = impute_zero(X_train, X_val, X_test, binary_cols_zero)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, target_encoder, numeric_imputer, knn_imputer

# ============================================
# SECTION 3: MAIN TRAINING CODE
# ============================================

def main():
    # 1. Load Data 
    df = pd.read_csv("immovlan_cleaned_file.csv")

    # 2. Clean data 
    columns_to_drop = [
    'url', 'Property ID', 'State of the property', 'Availability',
    'Livable surface', 'Furnished', 'Attic', 'Number of garages',
    'Kitchen equipment', 'Kitchen type', 'Number of showers',
    'Type of heating', 'Type of glazing', 'Number of facades',
    'Surface garden', 'Terrace', 'Surface terrace', 'Total land surface',
    'city', 'province', 'price_per_sqm', 'Price_per_sqm_land'
    ]

    df_clean = df.drop(columns_to_drop, axis='columns')

    # Drop rows where Price is missing (only ~3.5% of data)
    df_clean = df_clean.dropna(subset=['Price'])

    # Drop rows with unrealistic values (>20 bedrooms, bathrooms, or toilets)
    df_clean = df_clean[
        ~((df_clean["Number of bedrooms"] > 20) |
            (df_clean["Number of bathrooms"] > 20) |
            (df_clean["Number of toilets"] > 20))
    ]

    # 3. Define columns
    TARGET = 'Price'
    NUMERIC_COLS = ['Number of bedrooms', 'Number of bathrooms', 'Number of toilets']
    CATEGORICAL_COLS = ['type', 'Region']
    BINARY_COLS_KNN = ['Elevator', 'Garden']
    BINARY_COLS_ZERO = ['Garage', 'Swimming pool']

    # 4. Run preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, target_encoder, numeric_imputer, knn_imputer = preprocess(
        df=df_clean,
        target=TARGET,
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        binary_cols_knn=BINARY_COLS_KNN,
        binary_cols_zero=BINARY_COLS_ZERO
    )

    # 5. Train model
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # 6. Print score (optional)
    print(f"Validation R²: {xgb_model.score(X_val, y_val):.4f}")
    
    # 7. Save everything
    joblib.dump(xgb_model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(target_encoder, 'target_encoder.joblib')
    joblib.dump(numeric_imputer, 'numeric_imputer.joblib')
    joblib.dump(knn_imputer, 'knn_imputer.joblib')
    
    print("Model and preprocessing objects saved!")

# ============================================
# SECTION 4: RUN
# ============================================
if __name__ == "__main__":
    main()