import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from collections import Counter # Diperlukan oleh fungsi predict_random_forest_with_confidence

def predict_tree(tree_node, x_row):

    if tree_node.get('is_leaf', False):
        return tree_node['leaf_value']
    feature_idx = tree_node.get('feature_idx')
    threshold = tree_node.get('threshold')
    if x_row[feature_idx] <= threshold:
        return predict_tree(tree_node['left'], x_row)
    else:
        return predict_tree(tree_node['right'], x_row)

def predict_random_forest_with_confidence(list_of_trees, X_input_data):

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
Aplikasi ini menggunakan model Random Forest yang dibangun untuk merekomendasikan
tanaman yang cocok berdasarkan kondisi tanah dan lingkungan. Silakan geser slider di bawah ini:
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameter Tanah:")
    # Nama variabel (N, P, K, ph) harus sama dengan yang digunakan saat membuat input_features
    N = st.slider(
        'Kadar Nitrogen (N) (kg/ha)', 
        min_value=0, 
        max_value=150,  # Sesuaikan max_value berdasarkan data trainingmu
        value=90,       # Nilai default awal
        step=1          # Kenaikan per geseran
    )
    P = st.slider(
        'Kadar Fosfor (P) (kg/ha)', 
        min_value=0, 
        max_value=150,  # Sesuaikan max_value
        value=45, 
        step=1
    )
    K = st.slider(
        'Kadar Kalium (K) (kg/ha)', 
        min_value=0, 
        max_value=210,  # Sesuaikan max_value
        value=45, 
        step=1
    )
    ph = st.slider(
        'Tingkat pH Tanah', 
        min_value=3.0,  # Sesuaikan min_value dan max_value
        max_value=10.0, 
        value=6.5, 
        step=0.1,       # Step untuk nilai desimal
        format="%.1f"   # Format tampilan nilai slider
    )

with col2:
    st.subheader("Parameter Lingkungan:")
    temperature = st.slider(
        'Suhu (°C)', 
        min_value=5.0,   # Sesuaikan min_value dan max_value
        max_value=45.0, 
        value=25.0, 
        step=0.5,
        format="%.1f"
    )
    humidity = st.slider(
        'Kelembapan (%)', 
        min_value=10.0,  # Sesuaikan min_value dan max_value
        max_value=100.0, 
        value=70.0, 
        step=1.0,
        format="%.1f"
    )
    rainfall = st.slider(
        'Curah Hujan (mm)', 
        min_value=20.0,  # Sesuaikan min_value dan max_value
        max_value=300.0, 
        value=100.0, 
        step=5.0,        # Step bisa lebih besar untuk curah hujan
        format="%.1f"
    )

# Tombol untuk membuat prediksi (logika prediksi tetap sama)
if st.button('💡 Dapatkan Rekomendasi Tanaman', key="predict_button_slider"): # Ganti key jika perlu
    if loaded_rf_model_scratch is not None and loaded_label_encoder is not None:
        # Kumpulkan input dalam urutan yang benar
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
        
        try:
            numeric_prediction, confidence = predict_random_forest_with_confidence(loaded_rf_model_scratch, input_features)
            crop_name_prediction = loaded_label_encoder.inverse_transform(numeric_prediction)

            st.markdown("---")
            st.subheader("✔️ Rekomendasi Untuk Anda:")
            
            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.metric(label="Tanaman Direkomendasikan", value=crop_name_prediction[0])
            with col_conf:
                st.metric(label="confidence", value=f"{confidence[0]:.2f}%")
            
            # Hapus atau sesuaikan bagian emoji ini jika diinginkan
            # if confidence[0] > 75:
            #     st.balloons()
            # elif confidence[0] > 50:
            #     st.info("Rekomendasi cukup baik.")
            # else:
            #     st.warning("Model kurang yakin dengan rekomendasi ini.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.error("Pastikan fungsi `predict_tree` dan `predict_random_forest_with_confidence` dari scratch sudah benar.")
    else:
        st.error("Model atau encoder tidak berhasil dimuat. Aplikasi tidak bisa melakukan prediksi.")

# ... (Bagian footer tetap sama) ...
st.markdown("---")
st.markdown("Proyek Akhir Mata Kuliah AI | Model Klasifikasi Random Forest dari Scratch")
st.markdown(f"Waktu saat ini: {pd.Timestamp.now(tz='Asia/Jakarta').strftime('%A, %d %B %Y, %H:%M:%S %Z')}") # WIB Time