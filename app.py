import streamlit as st
import numpy as np
import pandas as pd
# import pickle # Pickle tidak lagi dibutuhkan untuk model sklearn jika pakai joblib
import joblib
# from collections import Counter # Tidak lagi dibutuhkan

# --- BAGIAN 1: FUNGSI PREDIKSI DARI SCRATCH DIHAPUS ---
# Fungsi predict_tree dan predict_random_forest_with_confidence TIDAK DIPERLUKAN LAGI
# karena kita akan menggunakan metode bawaan model scikit-learn.

# --- BAGIAN 2: MUAT MODEL SCIKIT-LEARN DAN ENCODER ---
SKLEARN_MODEL_FILENAME = 'sklearn_rf_model.joblib'  # Nama file model .joblib scikit-learn
ENCODER_FILENAME = 'final_crop_label_encoder.joblib' # atau nama file encoder-mu

@st.cache_resource
def load_sklearn_model_from_file(model_path):
    try:
        model_data = joblib.load(model_path) # Gunakan joblib untuk model scikit-learn
        return model_data
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

@st.cache_resource
def load_encoder_from_file(encoder_path): # Fungsi ini tetap sama
    try:
        encoder_data = joblib.load(encoder_path)
        return encoder_data
    except FileNotFoundError:
        st.error(f"Error: File encoder '{encoder_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat encoder: {e}")
        return None

# Muat model scikit-learn dan encoder
loaded_sklearn_rf_model = load_sklearn_model_from_file(SKLEARN_MODEL_FILENAME)
loaded_label_encoder = load_encoder_from_file(ENCODER_FILENAME)

# --- BAGIAN 3: ANTARMUKA PENGGUNA STREAMLIT (Input UI tetap sama) ---
st.set_page_config(page_title="Rekomendasi Tanaman (Scikit-learn)", page_icon="🌿", layout="wide")
st.title("🌿 Sistem Rekomendasi Tanaman (Scikit-learn) 🌿")
st.markdown("""
Aplikasi ini menggunakan model Random Forest dari Scikit-learn untuk merekomendasikan
tanaman yang cocok. Masukkan parameter di bawah ini:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Parameter Tanah:")
    N = st.slider('Kadar Nitrogen (N) (kg/ha)', 0, 150, 90, 1, key="N_sl")
    P = st.slider('Kadar Fosfor (P) (kg/ha)', 0, 150, 45, 1, key="P_sl")
    K = st.slider('Kadar Kalium (K) (kg/ha)', 0, 210, 45, 1, key="K_sl")
    ph = st.slider('Tingkat pH Tanah', 3.0, 10.0, 6.5, 0.1, format="%.1f", key="ph_sl")

with col2:
    st.subheader("Parameter Lingkungan:")
    temperature = st.slider('Suhu (°C)', 5.0, 45.0, 25.0, 0.5, format="%.1f", key="temp_sl")
    humidity = st.slider('Kelembapan (%)', 10.0, 100.0, 70.0, 1.0, format="%.1f", key="hum_sl")
    rainfall = st.slider('Curah Hujan (mm)', 20.0, 300.0, 100.0, 5.0, format="%.1f", key="rain_sl")

if st.button('💡 Dapatkan Rekomendasi (Scikit-learn)', key="predict_sklearn_button"):
    if loaded_sklearn_rf_model is not None and loaded_label_encoder is not None:
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
        try:
            # Gunakan metode .predict() dari model scikit-learn
            numeric_prediction = loaded_sklearn_rf_model.predict(input_features)

            # Gunakan metode .predict_proba() untuk confidence
            probabilities = loaded_sklearn_rf_model.predict_proba(input_features)
            # Confidence adalah probabilitas maksimum dari kelas yang diprediksi
            confidence = np.max(probabilities, axis=1) * 100
            numeric_prediction = loaded_sklearn_rf_model.predict(input_features)
            probabilities = loaded_sklearn_rf_model.predict_proba(input_features)
            confidence = np.max(probabilities, axis=1) * 100

            # --- TAMBAHKAN DEBUGGING DI SINI ---
            # st.subheader("🔍 INFORMASI DEBUGGING:")
            # if loaded_label_encoder is not None:
            #     st.write("**Kelas yang diketahui oleh LabelEncoder (Streamlit):**")
            #     st.write(list(loaded_label_encoder.classes_)) # Ini HARUS nama tanaman string
            #     st.write(f"**Jumlah kelas yang diketahui encoder:** {len(loaded_label_encoder.classes_)}")
            # else:
            #     st.error("**DEBUG: LabelEncoder TIDAK DIMUAT!**")

            # st.write("**Prediksi Numerik Mentah dari Model (Streamlit):**")
            # st.write(numeric_prediction) # Ini HARUS array berisi satu angka, misal [18]
            # if isinstance(numeric_prediction, np.ndarray) and numeric_prediction.ndim == 1:
            #     st.write(f"**Nilai numerik tunggal yang diprediksi:** {numeric_prediction[0]}")
            # st.write("--- AKHIR DEBUGGING ---")
            # # --- SELESAI DEBUGGING ---

            # Baris yang berpotensi error:
            crop_name_prediction = numeric_prediction[0]
            crop_name_prediction = confidence

            st.markdown("---")
            st.subheader("✔️ Rekomendasi Untuk Anda (Model Scikit-learn):")

            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.metric(label="Tanaman Direkomendasikan", value=crop_name_prediction[0])
            with col_conf:
                st.metric(label="Tingkat Keyakinan (Probabilitas)", value=f"{confidence[0]:.2f}%")

            if confidence[0] > 75:
                st.balloons()

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi dengan model Scikit-learn: {e}")
    else:
        st.error("Model Scikit-learn atau encoder tidak berhasil dimuat.")

# ... (Footer tetap sama) ...
st.markdown("---")
st.markdown("Proyek Akhir Mata Kuliah AI | Model Klasifikasi Random Forest Scikit-learn")