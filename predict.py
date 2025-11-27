# ============================================
# SECTION 1: IMPORTS
# ============================================
import pandas as pd
import joblib

# ============================================
# SECTION 2: LOAD SAVED OBJECTS
# ============================================
MODEL_PATH = "/Users/astha/data/Data Analysis/Becode_Git/Git/immo-eliza-ML/Model/"

model = joblib.load(MODEL_PATH + "model.joblib")
scaler = joblib.load(MODEL_PATH + "scaler.joblib")
target_encoder = joblib.load(MODEL_PATH + "target_encoder.joblib")
numeric_imputer = joblib.load(MODEL_PATH + "numeric_imputer.joblib")
knn_imputer = joblib.load(MODEL_PATH + "knn_imputer.joblib")
feature_order = joblib.load(MODEL_PATH + "feature_order.joblib")
metrics = joblib.load(MODEL_PATH + "metrics.joblib")

# ============================================
# SECTION 3: DEFINE COLUMN NAMES
# ============================================
NUMERIC_COLS = ['Number of bedrooms', 'Number of bathrooms', 'Number of toilets']
CATEGORICAL_COLS = ['type', 'Region']
BINARY_COLS_KNN = ['Elevator', 'Garden']
BINARY_COLS_ZERO = ['Garage', 'Swimming pool']

# ============================================
# SECTION 4: EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    
    # Example property
    new_property = {
        'Number of bedrooms': 3,
        'Number of bathrooms': 2,
        'Number of toilets': 1,
        'type': 'apartment',
        'Region': 'Brussels',
        'Elevator': 1,
        'Garden': 0,
        'Garage': 1,
        'Swimming pool': 0,
        'postal_code': 1000
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([new_property])
    
    # Apply preprocessing (same order as training)
    df[CATEGORICAL_COLS] = target_encoder.transform(df[CATEGORICAL_COLS])
    df[NUMERIC_COLS] = numeric_imputer.transform(df[NUMERIC_COLS])
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    df[BINARY_COLS_KNN] = knn_imputer.transform(df[BINARY_COLS_KNN])
    df[BINARY_COLS_ZERO] = df[BINARY_COLS_ZERO].fillna(0)
    
    # Reorder columns to match training
    df = df[feature_order]
    
    # Predict
    price = model.predict(df)[0]
    
    # Display result
    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"\n💰 Predicted Price: €{price:,.2f}")
    print(f"\n📊 Model Performance:")
    print(f"   R²:   {metrics['val_r2']:.4f}")
    print(f"   MAE:  ±€{metrics['val_mae']:,.0f}")
    print(f"   RMSE: ±€{metrics['val_rmse']:,.0f}")
    print(f"\n📍 Price Range (using MAE):")
    print(f"   Low:  €{price - metrics['val_mae']:,.2f}")
    print(f"   High: €{price + metrics['val_mae']:,.2f}")
    print("=" * 50)