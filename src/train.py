# src/train.py
#Testing Google Cloud Build triggered by pushing to GitHub repo

import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import warnings
import argparse
from google.cloud import storage


warnings.filterwarnings("ignore")

def simulate_data():
    dates = pd.date_range(start='2023-01-01', end='2025-09-30', freq='W-MON')
    n = len(dates)
    
    # Create a base trend and seasonality
    trend = np.linspace(50, 150, n)
    seasonality = 15 * np.sin(2 * np.pi * dates.isocalendar().week / 52) + 10 * np.sin(2 * np.pi * dates.isocalendar().week / 26)
    noise = np.random.normal(0, 10, n)
    
    # Base sales for a single SKU/size
    base_sales = trend + seasonality + noise
    
    df = pd.DataFrame({'date': dates, 'sales': base_sales})
    df['sales'] = df['sales'].astype(int).clip(lower=10) # Ensure sales are positive
    
    # Add external features
    df['price'] = np.random.uniform(25, 45, n).round(2)
    df['promotion_active'] = np.random.choice([0, 1], n, p=[0.8, 0.2])
    df['regional_weather_C'] = np.random.uniform(5, 20, n).round(1)
    
    # Promotions boost sales
    df.loc[df['promotion_active'] == 1, 'sales'] *= 1.2
    
    # Warmer weather boosts sales for this "dress"
    df.loc[df['regional_weather_C'] > 15, 'sales'] *= 1.1
    
    df['sales'] = df['sales'].astype(int)
    df = df.set_index('date')
    
    print("Simulated Data Head:")
    print(df.head())
    return df

def train_and_save_models(bucket_name, sarima_path, xgboost_path):
    """Trains and saves both SARIMA and XGBoost models."""
    df = simulate_data()
    
    # --- (Training code remains the same) ---
    print("Training SARIMA model...")
    order = (1, 1, 1)
    seasonal_order = (1, 1, 0, 52)
    sarima_model = SARIMAX(df['sales'], order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)

    print("Training XGBoost model...")
    df['sarima_forecast'] = sarima_fit.predict(start=df.index[0], end=df.index[-1])
    df['month'] = df.index.month
    df['week_of_year'] = df.index.isocalendar().week
    features = ['price', 'promotion_active', 'regional_weather_C', 'sarima_forecast', 'month', 'week_of_year']
    target = 'sales'
    X = df[features]
    y = df[target]
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror', n_jobs=-1)
    xgb_reg.fit(X, y)

    # --- Save models locally using the provided paths ---
    # ✅ FIX: Use the path variables instead of hardcoded strings
    print(f"Saving SARIMA model locally to: {sarima_path}")
    joblib.dump(sarima_fit, sarima_path)
    
    print(f"Saving XGBoost model locally to: {xgboost_path}")
    joblib.dump(xgb_reg, xgboost_path)

    # --- Upload to GCS ---
    print(f"Uploading models to GCS bucket: {bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # ✅ FIX: The blob path (destination) and local file path (source) should match
    blob_sarima = bucket.blob(sarima_path)
    blob_sarima.upload_from_filename(sarima_path)
    
    blob_xgboost = bucket.blob(xgboost_path)
    blob_xgboost.upload_from_filename(xgboost_path)

    print("Models successfully uploaded.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True, help='GCS bucket to save model artifacts')
    parser.add_argument('--sarima-path', type=str, default='models/sarima/sarima_model.pkl')
    parser.add_argument('--xgboost-path', type=str, default='models/xgboost/xgboost_model.pkl')
    args = parser.parse_args()
    
    train_and_save_models(args.bucket_name, args.sarima_path, args.xgboost_path)
