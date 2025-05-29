import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from collections import Counter # Diperlukan oleh fungsi predict_random_forest_with_confidence

# --- BAGIAN 1: DEFINISIKAN KEMBALI FUNGSI PREDIKSI DARI SCRATCH ---
# PENTING: Ganti `predict_random_forest` dengan `predict_random_forest_with_confidence`
# dan pastikan implementasi `predict_tree` juga ada di sini.

def predict_tree(tree_node, x_row):
    """
    SALIN IMPLEMENTASI FUNGSI predict_tree DARI SCRATCH MILIKMU KE SINI.
    (Sama seperti yang sudah kamu letakkan di app.py sebelumnya)
    """
    if tree_node.get('is_leaf', False):
        return tree_node['leaf_value']
    feature_idx = tree_node.get('feature_idx')
    threshold = tree_node.get('threshold')
    if x_row[feature_idx] <= threshold:
        return predict_tree(tree_node['left'], x_row)
    else:
        return predict_tree(tree_node['right'], x_row)

def predict_random_forest_with_confidence(list_of_trees, X_input_data):
    """
    SALIN IMPLEMENTASI FUNGSI predict_random_forest_with_confidence YANG BARU (DARI ATAS) KE SINI.
    """
    if X_input_data.ndim == 1:
        X_input_data = X_input_data.reshape(1, -1)
    n_samples = X_input_data.shape[0]
    num_trees = len(list_of_trees)
    all_individual_tree_predictions = np.zeros((n_samples, num_trees), dtype=int)
    for i, individual_tree in enumerate(list_of_trees):
        for j in range(n_samples):
            all_individual_tree_predictions[j, i] = predict_tree(individual_tree, X_input_data[j, :])
    final_aggregated_predictions = np.zeros(n_samples, dtype=int)
    confidence_scores = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        votes = Counter(all_individual_tree_predictions[i, :])
        if not votes: # Handle kasus jika tidak ada vote (meskipun seharusnya tidak terjadi dengan pohon yang valid)
            final_aggregated_predictions[i] = -1 # atau nilai default lain
            confidence_scores[i] = 0.0
            continue
        most_common_prediction_item = votes.most_common(1)[0]
        predicted_class = most_common_prediction_item[0]
        num_votes_for_predicted_class = most_common_prediction_item[1]
        final_aggregated_predictions[i] = predicted_class
        if num_trees > 0:
            confidence_scores[i] = (num_votes_for_predicted_class / num_trees) * 100
        else:
            confidence_scores[i] = 0.0
    return final_aggregated_predictions, confidence_scores

# --- BAGIAN 2: MUAT MODEL DAN ENCODER (Sama seperti sebelumnya) ---
MODEL_FILENAME = 'rf_scratch_model.pkl'
ENCODER_FILENAME = 'crop_label_encoder_for_scratch.joblib'

@st.cache_resource
def load_model_from_file(model_path):
    try:
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

@st.cache_resource
def load_encoder_from_file(encoder_path):
    try:
        encoder_data = joblib.load(encoder_path)
        return encoder_data
    except FileNotFoundError:
        st.error(f"Error: File encoder '{encoder_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat encoder: {e}")
        return None

loaded_rf_model_scratch = load_model_from_file(MODEL_FILENAME)
loaded_label_encoder = load_encoder_from_file(ENCODER_FILENAME)

# --- BAGIAN 3: ANTARMUKA PENGGUNA STREAMLIT (Input sama, output dimodifikasi) ---
st.set_page_config(page_title="Rekomendasi Tanaman", page_icon="🌿", layout="wide")

st.title("🌿 Sistem Rekomendasi Tanaman Cerdas 🌿")
st.markdown("""
Aplikasi ini menggunakan model Random Forest yang dibangun "dari scratch" untuk merekomendasikan
tanaman yang cocok berdasarkan kondisi tanah dan lingkungan. Masukkan parameter di bawah ini:
""")

# FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'] # Pastikan ini ada

col1, col2 = st.columns(2)
with col1:
    st.subheader("Parameter Tanah:")
    N = st.number_input('Kadar Nitrogen (N) (kg/ha)', min_value=0, max_value=200, value=90, key="N_input")
    P = st.number_input('Kadar Fosfor (P) (kg/ha)', min_value=0, max_value=150, value=40, key="P_input")
    K = st.number_input('Kadar Kalium (K) (kg/ha)', min_value=0, max_value=210, value=40, key="K_input")
    ph = st.number_input('Tingkat pH Tanah', min_value=0.0, max_value=14.0, value=6.5, format="%.2f", key="ph_input")

with col2:
    st.subheader("Parameter Lingkungan:")
    temperature = st.number_input('Suhu (°C)', min_value=-10.0, max_value=60.0, value=25.0, format="%.2f", key="temp_input")
    humidity = st.number_input('Kelembapan (%)', min_value=0.0, max_value=100.0, value=70.0, format="%.2f", key="hum_input")
    rainfall = st.number_input('Curah Hujan (mm)', min_value=0.0, max_value=350.0, value=150.0, format="%.2f", key="rain_input")

if st.button('💡 Dapatkan Rekomendasi Tanaman', key="predict_button"):
    if loaded_rf_model_scratch is not None and loaded_label_encoder is not None:
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
        try:
            # Gunakan fungsi yang sudah dimodifikasi
            numeric_prediction, confidence = predict_random_forest_with_confidence(loaded_rf_model_scratch, input_features)

            crop_name_prediction = loaded_label_encoder.inverse_transform(numeric_prediction)

            st.markdown("---")
            st.subheader("✔️ Rekomendasi Untuk Anda:")
            
            # Cara 1: Menggunakan st.metric untuk tampilan yang lebih menonjol
            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.metric(label="Tanaman Direkomendasikan", value=crop_name_prediction[0])
            with col_conf:
                st.metric(label="Tingkat Keyakinan (Voting)", value=f"{confidence[0]:.2f}%")

            # Cara 2: Menggunakan st.success dengan format lebih besar (opsional, bisa dikombinasikan atau pilih salah satu)
            # st.success(f"## **Tanaman yang paling cocok adalah: {crop_name_prediction[0]}**")
            # st.info(f"**Tingkat Keyakinan (Persentase Voting): {confidence[0]:.2f}%**")

            # Cara 3: Menggunakan Markdown untuk kustomisasi (lebih advanced jika ingin styling CSS)
            # st.markdown(f"""
            # <div style="background-color: #28a745; color: white; padding: 15px; border-radius: 7px; text-align: center;">
            #     <h2>Rekomendasi: {crop_name_prediction[0]}</h2>
            #     <p style="font-size: 1.2em;">Tingkat Keyakinan (Voting): {confidence[0]:.2f}%</p>
            # </div>
            # """, unsafe_allow_html=True)

            if confidence[0] > 75: # Beri emoji berdasarkan confidence
                st.balloons()
            elif confidence[0] > 50:
                st.success("👍")
            else:
                st.warning("🤔 Model kurang yakin dengan rekomendasi ini, pertimbangkan variasi input.")


        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
    else:
        st.error("Model atau encoder tidak berhasil dimuat. Aplikasi tidak bisa melakukan prediksi.")

st.markdown("---")
st.markdown("Proyek Akhir Mata Kuliah AI | Model Klasifikasi Random Forest dari Scratch")
st.markdown(f"Waktu saat ini: {pd.Timestamp.now(tz='Asia/Jakarta').strftime('%A, %d %B %Y, %H:%M:%S %Z')}") # WIB Time