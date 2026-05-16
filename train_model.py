# -*- coding: utf-8 -*-
"""
Script ini HANYA untuk generate file model (.pkl) agar Flask app bisa jalan.
Untuk versi lengkap dengan chart & analisis, gunakan Google_Colab_LinearRegression.py di Colab.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - tidak perlu display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

print("Memuat data...")
df = pd.read_csv('Car_sales.csv')
print(f"Shape awal: {df.shape}")

# === BERSIHKAN DATA ===
threshold = len(df.columns) * 0.5
df = df.dropna(thresh=threshold)

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.dropna(subset=['Price_in_thousands', 'Sales_in_thousands'])
print(f"Shape setelah cleaning: {df.shape}")

# === FEATURE ENGINEERING ===
df_model = pd.get_dummies(df, columns=['Manufacturer', 'Vehicle_type'], drop_first=False)

feature_cols_base = ['Sales_in_thousands', '__year_resale_value', 'Engine_size', 'Horsepower',
                     'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency']

ohe_cols = [c for c in df_model.columns if c.startswith('Manufacturer_') or c.startswith('Vehicle_type_')]
feature_cols = [c for c in (feature_cols_base + ohe_cols) if c in df_model.columns]
target_col = 'Price_in_thousands'

X = df_model[feature_cols]
y = df_model[target_col]

# === SPLIT & SCALE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# === TRAIN MODEL ===
print("Training Linear Regression...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# === EVALUASI ===
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# === SIMPAN MODEL ===
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler_lr.pkl')
joblib.dump({'feature_cols': feature_cols, 'feature_cols_base': feature_cols_base, 'ohe_cols': ohe_cols}, 'feature_info.pkl')

print("Model berhasil disimpan!")
print("  -> linear_regression_model.pkl")
print("  -> scaler_lr.pkl")
print("  -> feature_info.pkl")
print(f"\nR2 Score: {r2:.4f} | RMSE: {rmse:.4f}")
