import streamlit as st
import numpy as np
import joblib
import random

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Rekomendasi Tanaman Berbasis Random Forest",
    page_icon="🌿",
    layout="wide"
)

# Nama file model
SKLEARN_MODEL_FILENAME = 'sklearn_rf_model.joblib'

# Fungsi untuk memuat model
@st.cache_resource
def load_sklearn_model_from_file(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ File model '{model_path}' tidak ditemukan.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
    return None

# Muat model
loaded_sklearn_rf_model = load_sklearn_model_from_file(SKLEARN_MODEL_FILENAME)

# Inisialisasi session_state jika belum ada
default_values = {
    "N": 90,
    "P": 45,
    "K": 45,
    "ph": 6.5,
    "temperature": 25.0,
    "humidity": 70.0,
    "rainfall": 100.0,
}
for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Header aplikasi
st.title("🌿 Sistem Rekomendasi Tanaman Berbasis Random Forest 🌿")
st.markdown("""
Masukkan parameter karakteristik tanah dan lingkungan di bawah ini untuk mendapatkan rekomendasi tanaman yang sesuai.
""")

# UI input parameter
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameter Tanah:")
    st.slider('Kadar Nitrogen (N) (kg/ha)', 0, 150, key="N")
    st.slider('Kadar Fosfor (P) (kg/ha)', 0, 150, key="P")
    st.slider('Kadar Kalium (K) (kg/ha)', 0, 210, key="K")
    st.slider('Tingkat pH Tanah', 3.0, 10.0, key="ph", format="%.1f")

with col2:
    st.subheader("Parameter Lingkungan:")
    st.slider('Suhu (°C)', 5.0, 45.0, key="temperature", format="%.1f")
    st.slider('Kelembapan (%)', 10.0, 100.0, key="humidity", format="%.1f")
    st.slider('Curah Hujan (mm)', 20.0, 300.0, key="rainfall", format="%.1f")

# Dataset referensi
example_data = [
    [94, 53, 40, 20.28, 82.89, 5.72, 241.97, 'rice'],
    [74, 55, 19, 18.05, 62.89, 6.29, 84.24, 'maize'],
    [35, 66, 81, 19.37, 15.77, 6.14, 85.25, 'chickpea'],
    [37, 64, 22, 17.48, 18.83, 5.95, 121.94, 'kidneybeans'],
    [39, 77, 21, 23.00, 60.24, 4.60, 159.69, 'pigeonpeas'],
    [11, 44, 17, 26.34, 55.59, 8.02, 35.11, 'mothbeans'],
    [27, 56, 20, 29.21, 87.11, 6.42, 51.54, 'mungbean'],
    [58, 61, 15, 30.95, 64.23, 7.40, 62.79, 'blackgram'],
    [10, 56, 18, 28.00, 68.64, 7.33, 46.11, 'lentil'],
    [12, 29, 40, 19.68, 89.75, 6.59, 111.28, 'pomegranate'],
    [82, 78, 46, 25.06, 84.97, 5.74, 110.44, 'banana'],
    [26, 37, 30, 35.40, 49.46, 6.17, 97.41, 'mango'],
    [33, 120, 205, 35.12, 82.27, 5.55, 69.72, 'grapes'],
    [92, 20, 55, 25.10, 87.53, 6.59, 59.27, 'watermelon'],
    [80, 18, 52, 27.87, 91.15, 6.48, 24.05, 'muskmelon'],
    [12, 129, 205, 22.36, 91.16, 6.12, 118.68, 'apple'],
    [15, 14, 8, 10.01, 90.22, 6.22, 119.39, 'orange'],
    [57, 57, 51, 39.02, 91.49, 6.99, 105.88, 'papaya'],
    [27, 10, 33, 27.81, 97.48, 6.47, 154.06, 'coconut'],
    [111, 39, 22, 22.60, 80.35, 6.14, 88.57, 'cotton'],
    [61, 41, 44, 24.37, 82.11, 6.54, 159.92, 'jute']
]

def generate_random_example():
    base = random.choice(example_data)
    def vary(val, percent=0.1): return round(val + val * random.uniform(-percent, percent), 2)
    return {
        'N': int(vary(base[0], 0.15)),
        'P': int(vary(base[1], 0.15)),
        'K': int(vary(base[2], 0.15)),
        'temperature': vary(base[3], 0.05),
        'humidity': vary(base[4], 0.05),
        'ph': vary(base[5], 0.05),
        'rainfall': vary(base[6], 0.10),
        'label': base[7]
    }

# Tombol generate contoh acak
if st.button("🎲 Generate Contoh Acak"):
    example = generate_random_example()
    for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
        st.session_state[key] = example[key]
    st.toast(f"Contoh acak untuk label **{example['label']}** dimuat ke slider.", icon="🧪")

# Tombol prediksi
if st.button('💡 Dapatkan Rekomendasi (Scikit-learn)'):
    if loaded_sklearn_rf_model is not None:
        try:
            input_features = np.array([[st.session_state["N"], st.session_state["P"], st.session_state["K"],
                                        st.session_state["temperature"], st.session_state["humidity"],
                                        st.session_state["ph"], st.session_state["rainfall"]]])
            prediction = loaded_sklearn_rf_model.predict(input_features)
            probabilities = loaded_sklearn_rf_model.predict_proba(input_features)
            confidence = np.max(probabilities, axis=1)[0] * 100
            crop = prediction[0]

            # Tampilkan hasil
            st.toast(f"✅ Tanaman direkomendasikan: **{crop}** (Keyakinan: {confidence:.2f}%)", icon="🌱")

            st.markdown("---")
            st.markdown(f"""
            <div style="background-color:#004d00;padding:20px;border-radius:10px">
                <h2 style="color:white;text-align:center;">🌾 Rekomendasi Utama: <span style="color:#FFD700;">{crop.upper()}</span></h2>
                <p style="color:white;text-align:center;font-size:18px;">
                    Tanaman ini adalah yang paling ideal untuk kondisi tanah dan lingkungan yang Anda masukkan.
                </p>
                <h4 style="color:#90EE90;text-align:center;">📊 Confidence: {confidence:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {e}")
    else:
        st.error("Model belum dimuat.")

# Footer
st.markdown("---")
st.markdown("📘 Proyek Akhir Mata Kuliah AI | Model Klasifikasi Random Forest Scikit-learn")
