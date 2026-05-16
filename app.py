import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Pastikan working directory sudah benar agar menemukan file model
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path       = os.path.join(base_dir, 'linear_regression_model.pkl')
scaler_path      = os.path.join(base_dir, 'scaler_lr.pkl')
feat_info_path   = os.path.join(base_dir, 'feature_info.pkl')

try:
    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    feature_info = joblib.load(feat_info_path)
    FEATURE_COLS      = feature_info['feature_cols']
    FEATURE_COLS_BASE = feature_info['feature_cols_base']
    print("[OK] Model Linear Regression berhasil dimuat!")
except Exception as e:
    print(f"[ERROR] Gagal memuat model/scaler: {e}")
    FEATURE_COLS      = []
    FEATURE_COLS_BASE = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Buat dataframe 1 baris berisi nilai 0 untuk semua kolom expected
        df = pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)

        # Isi nilai fitur numerik
        for col in FEATURE_COLS_BASE:
            if col in data and data[col] != '':
                df.at[0, col] = float(data[col])

        # Tangani One-Hot Encoding untuk Manufacturer
        manufacturer = data.get('Manufacturer', '')
        col_manuf = f"Manufacturer_{manufacturer}"
        if col_manuf in df.columns:
            df.at[0, col_manuf] = 1.0

        # Tangani One-Hot Encoding untuk Vehicle Type
        vehicle_type = data.get('Vehicle_type', '')
        col_vtype = f"Vehicle_type_{vehicle_type}"
        if col_vtype in df.columns:
            df.at[0, col_vtype] = 1.0

        # Lakukan Scaling
        X_scaled = scaler.transform(df)

        # Prediksi harga menggunakan Linear Regression
        pred = model.predict(X_scaled)[0]

        # Target adalah Price_in_thousands → kalikan 1000 untuk harga asli
        actual_price = pred * 1000

        return jsonify({
            'status': 'success',
            'predicted_price_thousands': round(pred, 2),
            'predicted_price_formatted': f"${actual_price:,.2f}"
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
