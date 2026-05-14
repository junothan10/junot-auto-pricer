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
model_path = os.path.join(base_dir, 'best_rf_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    EXPECTED_COLS = list(scaler.feature_names_in_)
except Exception as e:
    print(f"Gagal memuat model/scaler: {e}")
    EXPECTED_COLS = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Buat dataframe 1 baris berisi nilai 0 untuk semua kolom expected
        df = pd.DataFrame(np.zeros((1, len(EXPECTED_COLS))), columns=EXPECTED_COLS)
        
        # Isi nilai fitur numerik
        num_cols = ['Sales_in_thousands', '__year_resale_value', 'Engine_size', 'Horsepower', 
                    'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency']
        for col in num_cols:
            if col in data and data[col] != '':
                df.at[0, col] = float(data[col])
                
        # Tangani One-Hot Encoding untuk Manufacturer
        manufacturer = data.get('Manufacturer', '')
        col_manuf = f"Manufacturer_{manufacturer}"
        if col_manuf in df.columns:
            df.at[0, col_manuf] = 1.0
            
        # Tangani One-Hot Encoding untuk Vehicle Type
        vehicle_type = data.get('Vehicle_type', '')
        if vehicle_type == 'Passenger':
            df.at[0, 'Vehicle_type_Passenger'] = 1.0
            
        # Lakukan Scaling
        X_scaled = scaler.transform(df)
        
        # Prediksi harga
        pred = model.predict(X_scaled)[0]
        
        # Karena target adalah Price_in_thousands (Harga x 1000)
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
