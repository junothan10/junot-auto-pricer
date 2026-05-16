
# -*- coding: utf-8 -*-
# ============================================================
# FINAL PROJECT - DATA SCIENCE
# PREDIKSI HARGA MOBIL MENGGUNAKAN LINEAR REGRESSION
# CRISP-DM METHODOLOGY
# Nama  : Husni Ayi Nurdin
# NPM   : 237006002
# ============================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# LANGKAH 1: MEMANGGIL LIBRARY YANG DIPERLUKAN
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("✅ Semua library berhasil diimport!")

# ============================================================
# LANGKAH 2: LOAD DATA
# ============================================================
# Jika menggunakan Google Colab, upload file Car_sales.csv terlebih dahulu
# dari menu Files di sebelah kiri, lalu jalankan cell ini.

# Untuk Colab - uncomment baris di bawah jika perlu upload manual:
# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv('Car_sales.csv')
print(f"✅ Data berhasil dimuat! Shape: {df.shape}")

# ============================================================
# LANGKAH 3: MELIHAT DATA
# ============================================================
print("\n📋 Preview Data (5 Baris Pertama):")
print(df.head())

print("\n📊 Informasi Dataset:")
print(df.info())

print("\n📈 Statistik Deskriptif:")
print(df.describe())

print("\n🔍 Kolom-kolom dalam dataset:")
print(df.columns.tolist())

# ============================================================
# LANGKAH 4: MENGHAPUS MISSING VALUE
# ============================================================
print("\n\n🔎 Mengecek Missing Value sebelum penanganan:")
print(df.isnull().sum())

# Hapus baris yang memiliki terlalu banyak nilai kosong (threshold > 50%)
threshold = len(df.columns) * 0.5
df_cleaned = df.dropna(thresh=threshold)
print(f"\n✅ Baris setelah menghapus yang terlalu banyak NaN: {df_cleaned.shape[0]} baris")

# ============================================================
# LANGKAH 5: MENGISI MISSING VALUE
# ============================================================
print("\n🔧 Mengisi Missing Value...")

# Pisahkan kolom numerik dan kategorikal
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()

# Isi kolom numerik dengan median
for col in numeric_cols:
    if df_cleaned[col].isnull().sum() > 0:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
        print(f"  → Kolom '{col}' diisi dengan median: {median_val:.2f}")

# Isi kolom kategorikal dengan modus
for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        mode_val = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(mode_val)
        print(f"  → Kolom '{col}' diisi dengan modus: {mode_val}")

print("\n✅ Cek Missing Value setelah penanganan:")
print(df_cleaned.isnull().sum())
print(f"\nShape akhir data bersih: {df_cleaned.shape}")

# ============================================================
# LANGKAH 6: EKSPLORASI DATA
# ============================================================
print("\n\n📊 ===== EKSPLORASI DATA =====")

# Pastikan kolom Sales_in_thousands ada dan bersih
df_cleaned['Sales_in_thousands'] = pd.to_numeric(df_cleaned['Sales_in_thousands'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['Sales_in_thousands', 'Price_in_thousands'])

# Top 10 Mobil dengan Penjualan Terbanyak
top10_sales = df_cleaned.nlargest(10, 'Sales_in_thousands')[['Manufacturer', 'Model', 'Sales_in_thousands']].reset_index(drop=True)
top10_sales['Car_Name'] = top10_sales['Manufacturer'] + ' ' + top10_sales['Model']

print("\n🏆 10 Mobil dengan Penjualan Terbanyak:")
print(top10_sales[['Car_Name', 'Sales_in_thousands']])

# ============================================================
# LANGKAH 6a: Chart 10 Mobil dengan Penjualan Terbanyak
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, 10))[::-1]

bars = ax.barh(top10_sales['Car_Name'][::-1], top10_sales['Sales_in_thousands'][::-1], color=colors, edgecolor='white', height=0.65)

# Tambahkan label nilai
for bar, val in zip(bars, top10_sales['Sales_in_thousands'][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}k', va='center', fontsize=10, fontweight='bold', color='#333')

ax.set_title('🏆 Top 10 Mobil dengan Jumlah Penjualan Terbanyak', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Penjualan (dalam ribuan unit)', fontsize=11)
ax.set_ylabel('Nama Mobil', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f9f9f9')
plt.tight_layout()
plt.savefig('chart_top10_penjualan.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
📝 ANALISIS PENJUALAN:
- Grafik di atas menampilkan 10 mobil dengan volume penjualan tertinggi.
- Ford F-Series dan Chevrolet mendominasi penjualan karena popularitasnya sebagai kendaraan keluarga dan utilitas di Amerika.
- Kendaraan dari segmen sedan dan pickup mendominasi daftar teratas.
- Pabrikan lokal Amerika (Ford, Chevrolet, Dodge) mendominasi volume penjualan, 
  menunjukkan preferensi pasar domestik yang kuat.
""")

# ============================================================
# LANGKAH 6b: Harga dari 10 Mobil Terlaris
# ============================================================
top10_data = df_cleaned.nlargest(10, 'Sales_in_thousands').reset_index(drop=True)
top10_data['Car_Name'] = top10_data['Manufacturer'] + ' ' + top10_data['Model']

print("\n💰 Harga dari 10 Mobil Terlaris:")
print(top10_data[['Car_Name', 'Sales_in_thousands', 'Price_in_thousands']])

fig, ax = plt.subplots(figsize=(12, 6))
colors_price = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
bars2 = ax.bar(range(len(top10_data)), top10_data['Price_in_thousands'], color=colors_price, edgecolor='white', width=0.6)

for bar, val in zip(bars2, top10_data['Price_in_thousands']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'${val:.1f}k', ha='center', fontsize=9, fontweight='bold', color='#1a3a5c')

ax.set_xticks(range(len(top10_data)))
ax.set_xticklabels(top10_data['Car_Name'], rotation=40, ha='right', fontsize=9)
ax.set_title('💰 Harga dari 10 Mobil dengan Penjualan Terbanyak', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Harga (dalam ribuan USD)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f0f4f8')
plt.tight_layout()
plt.savefig('chart_top10_harga.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
📝 ANALISIS HARGA:
- Mobil terlaris tidak selalu yang termahal - ini menunjukkan bahwa harga yang terjangkau
  berpengaruh besar terhadap volume penjualan.
- Rentang harga mobil terlaris berkisar antara $15.000 - $35.000 (segmen menengah).
- Mobil premium seperti Cadillac memiliki harga tinggi namun volume lebih rendah dibanding Ford/Chevrolet.
""")

# ============================================================
# LANGKAH 6c: 3 Variabel Lain dari 10 Mobil Terlaris
# ============================================================
print("\n🔧 Atribut Lain dari 10 Mobil Terlaris:")

# Variable 1: Horsepower
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('📊 Atribut Tambahan dari 10 Mobil Terlaris', fontsize=14, fontweight='bold', y=1.02)

# Horsepower
axes[0].bar(range(len(top10_data)), top10_data['Horsepower'], color=plt.cm.Oranges(np.linspace(0.4, 0.9, 10)), edgecolor='white')
axes[0].set_xticks(range(len(top10_data)))
axes[0].set_xticklabels(top10_data['Car_Name'], rotation=45, ha='right', fontsize=8)
axes[0].set_title('🔥 Horsepower (HP)', fontweight='bold')
axes[0].set_ylabel('Horsepower')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for i, v in enumerate(top10_data['Horsepower']):
    axes[0].text(i, v + 1, f'{v:.0f}', ha='center', fontsize=8, fontweight='bold')

# Engine Size
axes[1].bar(range(len(top10_data)), top10_data['Engine_size'], color=plt.cm.Greens(np.linspace(0.4, 0.9, 10)), edgecolor='white')
axes[1].set_xticks(range(len(top10_data)))
axes[1].set_xticklabels(top10_data['Car_Name'], rotation=45, ha='right', fontsize=8)
axes[1].set_title('⚙️ Engine Size (Liter)', fontweight='bold')
axes[1].set_ylabel('Engine Size (L)')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
for i, v in enumerate(top10_data['Engine_size']):
    axes[1].text(i, v + 0.02, f'{v:.1f}L', ha='center', fontsize=8, fontweight='bold')

# Fuel Efficiency
axes[2].bar(range(len(top10_data)), top10_data['Fuel_efficiency'], color=plt.cm.Purples(np.linspace(0.4, 0.9, 10)), edgecolor='white')
axes[2].set_xticks(range(len(top10_data)))
axes[2].set_xticklabels(top10_data['Car_Name'], rotation=45, ha='right', fontsize=8)
axes[2].set_title('⛽ Fuel Efficiency (MPG)', fontweight='bold')
axes[2].set_ylabel('MPG')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
for i, v in enumerate(top10_data['Fuel_efficiency']):
    axes[2].text(i, v + 0.3, f'{v:.0f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('chart_top10_atribut_lain.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
📝 ANALISIS ATRIBUT TAMBAHAN:
1. HORSEPOWER: Mobil dengan penjualan tinggi cenderung memiliki horsepower menengah (120-200 HP).
   Mobil dengan HP terlalu besar cenderung lebih mahal dan kurang laku di pasar massal.
2. ENGINE SIZE: Mesin berukuran 2.0L - 4.0L mendominasi pasar. Mesin kecil lebih irit,
   mesin besar memberikan performa lebih tapi konsumsi BBM lebih boros.
3. FUEL EFFICIENCY: Efisiensi bahan bakar yang baik (20-35 MPG) menjadi daya tarik bagi 
   konsumen modern yang mempertimbangkan biaya operasional kendaraan jangka panjang.
""")

# ============================================================
# LANGKAH 7: MENENTUKAN VARIABEL REKOMENDASI SPESIFIKASI
# ============================================================
print("\n\n🎯 ===== LANGKAH 7: VARIABEL REKOMENDASI SPESIFIKASI =====")
print("""
Berdasarkan eksplorasi data yang telah dilakukan, variabel-variabel berikut
dipilih sebagai fitur untuk model prediksi harga mobil:

VARIABEL INDEPENDEN (X) / FITUR:
1.  Sales_in_thousands    → Volume penjualan mencerminkan popularitas di pasar
2.  __year_resale_value   → Nilai jual kembali menunjukkan kualitas & retensi nilai
3.  Engine_size           → Ukuran mesin berkorelasi langsung dengan harga & performa
4.  Horsepower            → Tenaga mesin merupakan faktor utama penentu harga
5.  Wheelbase             → Jarak sumbu roda menentukan kelas kendaraan (compact/full-size)
6.  Width                 → Lebar kendaraan mencerminkan ukuran & kelas segmen
7.  Length                → Panjang kendaraan berkaitan dengan kapasitas & kelas
8.  Curb_weight           → Bobot kendaraan berkaitan dengan material & fitur
9.  Fuel_capacity         → Kapasitas tangki berhubungan dengan ukuran kendaraan
10. Fuel_efficiency       → Efisiensi BBM penting bagi segmen pasar menengah
11. Manufacturer          → Merek/pabrikan sangat mempengaruhi harga (one-hot encoded)
12. Vehicle_type          → Jenis kendaraan (Passenger/Car) membedakan segmen pasar

VARIABEL DEPENDEN (Y) / TARGET:
- Price_in_thousands → Harga mobil dalam satuan ribuan USD

REKOMENDASI SPESIFIKASI UNTUK PASAR:
Berdasarkan analisis data, spesifikasi yang paling diminati pasar adalah:
- Engine Size  : 2.0L - 3.0L
- Horsepower   : 130 - 200 HP
- Fuel Eff.    : 25 - 35 MPG
- Price Range  : $15.000 - $30.000
- Vehicle Type : Passenger Car
""")

# ============================================================
# LANGKAH 8: MEMBUAT MODEL PREDIKSI HARGA - LINEAR REGRESSION
# ============================================================
print("\n\n🤖 ===== LANGKAH 8: MODEL LINEAR REGRESSION =====")

# Siapkan data
df_model = df_cleaned.copy()

# 8a. Memisahkan Variabel Independent dan Dependent
print("\n8a. Memisahkan Variabel Independent (X) dan Dependent (Y)...")

# One-Hot Encoding untuk variabel kategorikal
df_model = pd.get_dummies(df_model, columns=['Manufacturer', 'Vehicle_type'], drop_first=False)

# Pilih kolom fitur numerik asli
feature_cols_base = ['Sales_in_thousands', '__year_resale_value', 'Engine_size', 'Horsepower',
                     'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency']

# Tambahkan kolom hasil one-hot encoding
ohe_cols = [c for c in df_model.columns if c.startswith('Manufacturer_') or c.startswith('Vehicle_type_')]
feature_cols = feature_cols_base + ohe_cols

# Pastikan semua kolom ada
feature_cols = [c for c in feature_cols if c in df_model.columns]
target_col = 'Price_in_thousands'

X = df_model[feature_cols]
y = df_model[target_col]

print(f"  Jumlah fitur (X): {X.shape[1]} kolom")
print(f"  Jumlah sampel    : {X.shape[0]} baris")
print(f"  Target (y)       : {target_col}")

# 8b. Memisahkan Data Training 80% dan Testing 20%
print("\n8b. Split Data Training (80%) dan Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Data Training : {X_train.shape[0]} sampel")
print(f"  Data Testing  : {X_test.shape[0]} sampel")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 8c. Membuat Model Regresi Linear
print("\n8c. Membuat Model Linear Regression...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("  ✅ Model Linear Regression berhasil dilatih!")

# 8d. Evaluasi Model dengan RMSE dan R2 Score
print("\n8d. Evaluasi Model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test, y_pred_test)

print(f"\n{'='*45}")
print(f"{'  HASIL EVALUASI MODEL LINEAR REGRESSION':^45}")
print(f"{'='*45}")
print(f"  {'Metrik':<20} {'Training':>10} {'Testing':>10}")
print(f"  {'-'*40}")
print(f"  {'RMSE':<20} {rmse_train:>10.4f} {rmse_test:>10.4f}")
print(f"  {'R2 Score':<20} {r2_train:>10.4f} {r2_test:>10.4f}")
print(f"{'='*45}")

print(f"""
📝 INTERPRETASI HASIL:
- RMSE Testing = {rmse_test:.4f} → Rata-rata error prediksi harga sekitar ${rmse_test*1000:,.0f}
- R² Score Testing = {r2_test:.4f} → Model mampu menjelaskan {r2_test*100:.1f}% variansi harga
- Semakin mendekati 1.0, semakin baik model dalam memprediksi harga
""")

# 8e. Scatter Plot Evaluasi
print("\n8e. Menggambar Scatter Plot Hasil Evaluasi...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('📈 Evaluasi Model Linear Regression - Harga Aktual vs Prediksi', fontsize=14, fontweight='bold')

# Plot Training
axes[0].scatter(y_train, y_pred_train, alpha=0.6, color='steelblue', edgecolors='white', s=60, label='Data Points')
min_val = min(y_train.min(), y_pred_train.min())
max_val = max(y_train.max(), y_pred_train.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_title(f'Data Training\nR² = {r2_train:.4f} | RMSE = {rmse_train:.4f}', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Harga Aktual (thousands USD)', fontsize=10)
axes[0].set_ylabel('Harga Prediksi (thousands USD)', fontsize=10)
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_facecolor('#f0f4f8')

# Plot Testing
axes[1].scatter(y_test, y_pred_test, alpha=0.7, color='#e67e22', edgecolors='white', s=60, label='Data Points')
min_val2 = min(y_test.min(), y_pred_test.min())
max_val2 = max(y_test.max(), y_pred_test.max())
axes[1].plot([min_val2, max_val2], [min_val2, max_val2], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title(f'Data Testing\nR² = {r2_test:.4f} | RMSE = {rmse_test:.4f}', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Harga Aktual (thousands USD)', fontsize=10)
axes[1].set_ylabel('Harga Prediksi (thousands USD)', fontsize=10)
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_facecolor('#f0f4f8')

plt.tight_layout()
plt.savefig('scatter_plot_evaluasi.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
📝 INTERPRETASI SCATTER PLOT:
- Titik-titik yang mendekati garis merah putus-putus (y=x) menunjukkan prediksi yang akurat.
- Garis putus-putus merah = garis "perfect prediction" (prediksi sempurna).
- Jika R² mendekati 1.0 dan titik-titik rapat di sekitar garis, model dikatakan baik.
""")

# ============================================================
# SIMPAN MODEL DAN SCALER
# ============================================================
print("\n\n💾 Menyimpan model dan scaler...")
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler_lr.pkl')

# Simpan juga daftar kolom untuk digunakan saat prediksi
feature_info = {
    'feature_cols': feature_cols,
    'feature_cols_base': feature_cols_base,
    'ohe_cols': ohe_cols
}
joblib.dump(feature_info, 'feature_info.pkl')

print("  ✅ Model disimpan: linear_regression_model.pkl")
print("  ✅ Scaler disimpan: scaler_lr.pkl")
print("  ✅ Feature info disimpan: feature_info.pkl")

# ============================================================
# LANGKAH 9: MEMPREDIKSI HARGA DENGAN SPESIFIKASI TERTENTU
# ============================================================
print("\n\n🔮 ===== LANGKAH 9: PREDIKSI HARGA DENGAN DATAFRAME =====")
print("""
Rekomendasi Spesifikasi Mobil Berdasarkan Analisis Pasar:
- Manufacturer : Ford (terlaris di pasar)
- Vehicle Type : Passenger
- Engine Size  : 2.5L (efisien & bertenaga)
- Horsepower   : 150 HP
- Fuel Efficiency: 28 MPG
- Wheelbase    : 105 inch
""")

# Buat DataFrame spesifikasi yang ingin diprediksi
spec_input = {
    'Sales_in_thousands'  : 50.0,
    '__year_resale_value' : 18.0,
    'Engine_size'         : 2.5,
    'Horsepower'          : 150.0,
    'Wheelbase'           : 105.0,
    'Width'               : 68.0,
    'Length'              : 175.0,
    'Curb_weight'         : 3.0,
    'Fuel_capacity'       : 14.5,
    'Fuel_efficiency'     : 28.0,
    'Manufacturer'        : 'Ford',
    'Vehicle_type'        : 'Passenger'
}

df_spec = pd.DataFrame([spec_input])
print("\n📋 Spesifikasi Input:")
print(df_spec.to_string(index=False))

# Proses OHE untuk prediksi
df_pred = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

# Isi nilai numerik
for col in feature_cols_base:
    if col in df_spec.columns:
        df_pred.at[0, col] = float(df_spec[col].values[0])

# Isi OHE Manufacturer
manuf_col = f"Manufacturer_{spec_input['Manufacturer']}"
if manuf_col in df_pred.columns:
    df_pred.at[0, manuf_col] = 1.0

# Isi OHE Vehicle_type
vtype_col = f"Vehicle_type_{spec_input['Vehicle_type']}"
if vtype_col in df_pred.columns:
    df_pred.at[0, vtype_col] = 1.0

# Scaling
df_pred_scaled = scaler.transform(df_pred)

# Prediksi
predicted_price = model.predict(df_pred_scaled)[0]
predicted_price_usd = predicted_price * 1000

# ============================================================
# LANGKAH 10: MENAMPILKAN HASIL PREDIKSI
# ============================================================
print("\n\n💰 ===== LANGKAH 10: HASIL PREDIKSI HARGA =====")
print(f"""
╔══════════════════════════════════════════════════════════╗
║           HASIL PREDIKSI HARGA MOBIL                   ║
╠══════════════════════════════════════════════════════════╣
║  Manufacturer  : {spec_input['Manufacturer']:<38} ║
║  Vehicle Type  : {spec_input['Vehicle_type']:<38} ║
║  Engine Size   : {spec_input['Engine_size']} L{' '*36} ║
║  Horsepower    : {spec_input['Horsepower']} HP{' '*35} ║
║  Fuel Effic.   : {spec_input['Fuel_efficiency']} MPG{' '*34} ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  💵 PREDIKSI HARGA  : ${predicted_price:.2f} ribu (${predicted_price_usd:,.2f}){' '*5} ║
║                                                        ║
╚══════════════════════════════════════════════════════════╝
""")

print("\n✅ SELESAI! Model Linear Regression berhasil dilatih dan digunakan untuk prediksi.")
print(f"\nFile yang dihasilkan:")
print(f"  1. linear_regression_model.pkl  → Model Linear Regression")
print(f"  2. scaler_lr.pkl                → Standard Scaler")
print(f"  3. feature_info.pkl             → Informasi Fitur")
print(f"  4. chart_top10_penjualan.png    → Chart 6a")
print(f"  5. chart_top10_harga.png        → Chart 6b")
print(f"  6. chart_top10_atribut_lain.png → Chart 6c")
print(f"  7. scatter_plot_evaluasi.png    → Scatter Plot Evaluasi")
