import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Sistem Rekomendasi Tanaman Berbasis Random Forest", page_icon="🌿", layout="wide")

# Nama file model
SKLEARN_MODEL_FILENAME = 'sklearn_rf_model.joblib'

@st.cache_resource
def load_sklearn_model_from_file(model_path):
    try:
        model_data = joblib.load(model_path)
        return model_data
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Muat model
loaded_sklearn_rf_model = load_sklearn_model_from_file(SKLEARN_MODEL_FILENAME)

# UI
st.title("🌿 Sistem Rekomendasi Tanaman Berbasis Random Forest 🌿")
st.markdown("""
Aplikasi ini menggunakan model Random Forest untuk merekomendasikan
tanaman yang cocok sesuai karakteristik tanah. Masukkan parameter di bawah ini:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Parameter Tanah:")
    N = st.slider('Kadar Nitrogen (N) (kg/ha)', 0, 150, 90, 1)
    P = st.slider('Kadar Fosfor (P) (kg/ha)', 0, 150, 45, 1)
    K = st.slider('Kadar Kalium (K) (kg/ha)', 0, 210, 45, 1)
    ph = st.slider('Tingkat pH Tanah', 3.0, 10.0, 6.5, 0.1, format="%.1f")

with col2:
    st.subheader("Parameter Lingkungan:")
    temperature = st.slider('Suhu (°C)', 5.0, 45.0, 25.0, 0.5, format="%.1f")
    humidity = st.slider('Kelembapan (%)', 10.0, 100.0, 70.0, 1.0, format="%.1f")
    rainfall = st.slider('Curah Hujan (mm)', 20.0, 300.0, 100.0, 5.0, format="%.1f")

if st.button('💡 Dapatkan Rekomendasi (Scikit-learn)'):
    if loaded_sklearn_rf_model is not None:
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
        try:
            prediction = loaded_sklearn_rf_model.predict(input_features)
            probabilities = loaded_sklearn_rf_model.predict_proba(input_features)
            confidence = np.max(probabilities, axis=1) * 100
            crop = prediction[0]
            conf = confidence[0]

            # st.markdown("---")
            # st.subheader("✔️ Rekomendasi Untuk Anda (Model Scikit-learn):")

            # col_pred, col_conf = st.columns(2)
            # with col_pred:
            #     st.metric(label="Tanaman Direkomendasikan", value=prediction[0])
            # with col_conf:
            #     st.metric(label="Tingkat Keyakinan (Probabilitas)", value=f"{confidence[0]:.2f}%")

            # if confidence[0] > 75:
            #     st.balloons()

            st.toast(f"✅ Tanaman direkomendasikan: **{crop}** (Keyakinan: {conf:.2f}%)", icon="🌱")

            # Blok hasil utama
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color:#004d00;padding:20px;border-radius:10px">
                <h2 style="color:white;text-align:center;">🌾 Rekomendasi Utama: <span style="color:#FFD700;">{crop.upper()}</span></h2>
                <p style="color:white;text-align:center;font-size:18px;">
                    Tanaman ini adalah yang paling cocok untuk kondisi tanah dan lingkungan yang Anda masukkan.
                </p>
                <h4 style="color:#90EE90;text-align:center;">📊 Keyakinan Model: {conf:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)


        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            import traceback
            st.text(traceback.format_exc())
    else:
        st.error("Model Scikit-learn tidak berhasil dimuat.")

st.markdown("---")
st.markdown("Proyek Akhir Mata Kuliah AI | Model Klasifikasi Random Forest Scikit-learn")
