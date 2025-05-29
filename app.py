import streamlit as st
st.set_page_config(page_title="Rekomendasi Tanaman", page_icon="🌿", layout="wide")
import numpy as np
import pickle
import joblib
from collections import Counter 


def predict_tree(tree_node, x_row):

    if tree_node.get('is_leaf', False): 
        return tree_node['leaf_value']

    feature_idx = tree_node.get('feature_idx')
    threshold = tree_node.get('threshold')

    if feature_idx is None or threshold is None:
        
        st.error("Error pada struktur pohon: node internal tanpa fitur/threshold.")

        pass # Hapus pass dan pastikan logic tree traversalmu benar

    if x_row[feature_idx] <= threshold:
        return predict_tree(tree_node['left'], x_row)
    else:
        return predict_tree(tree_node['right'], x_row)

def predict_random_forest(list_of_trees, X_input_data):
   
    if X_input_data.ndim == 1:
        X_input_data = X_input_data.reshape(1, -1)

    n_samples = X_input_data.shape[0]
    all_individual_tree_predictions = np.zeros((n_samples, len(list_of_trees)), dtype=int)

    for i, individual_tree in enumerate(list_of_trees):
        for j in range(n_samples):
            all_individual_tree_predictions[j, i] = predict_tree(individual_tree, X_input_data[j, :])

    final_aggregated_predictions = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Mayoritas voting
        most_common_prediction = Counter(all_individual_tree_predictions[i, :]).most_common(1)[0][0]
        final_aggregated_predictions[i] = most_common_prediction

    return final_aggregated_predictions

MODEL_FILENAME = 'rf_scratch_model.pkl'  
ENCODER_FILENAME = 'crop_label_encoder_for_scratch.joblib' 

# Menggunakan @st.cache_resource untuk memuat model
@st.cache_resource
def load_model_from_file(model_path):
    try:
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan. Pastikan file ada di direktori yang benar.")
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
        st.error(f"Error: File encoder '{encoder_path}' tidak ditemukan. Pastikan file ada di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat encoder: {e}")
        return None

# Muat model dan encoder
loaded_rf_model_scratch = load_model_from_file(MODEL_FILENAME)
loaded_label_encoder = load_encoder_from_file(ENCODER_FILENAME)



st.title("🌿 Sistem Rekomendasi Tanaman Cerdas 🌿")
st.markdown("""
Aplikasi ini menggunakan model Random Forest yang dibangun untuk merekomendasikan
tanaman yang cocok berdasarkan kondisi tanah dan lingkungan. Masukkan parameter di bawah ini:
""")


FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Buat kolom untuk layout yang lebih rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameter Tanah:")
    N = st.number_input('Kadar Nitrogen (N) (kg/ha)', min_value=0, max_value=200, value=90)
    P = st.number_input('Kadar Fosfor (P) (kg/ha)', min_value=0, max_value=150, value=40)
    K = st.number_input('Kadar Kalium (K) (kg/ha)', min_value=0, max_value=210, value=40)
    ph = st.number_input('Tingkat pH Tanah', min_value=0.0, max_value=14.0, value=6.5, format="%.2f")

with col2:
    st.subheader("Parameter Lingkungan:")
    temperature = st.number_input('Suhu (°C)', min_value=-10.0, max_value=60.0, value=25.0, format="%.2f")
    humidity = st.number_input('Kelembapan (%)', min_value=0.0, max_value=100.0, value=70.0, format="%.2f")
    rainfall = st.number_input('Curah Hujan (mm)', min_value=0.0, max_value=350.0, value=150.0, format="%.2f")

# Tombol untuk membuat prediksi
if st.button('💡 Dapatkan Rekomendasi Tanaman'):
    if loaded_rf_model_scratch is not None and loaded_label_encoder is not None:
        # Kumpulkan input dalam urutan yang benar
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)


        try:
            # Buat prediksi numerik menggunakan model dari scratch
            numeric_prediction = predict_random_forest(loaded_rf_model_scratch, input_features)

            # Ubah prediksi numerik menjadi nama tanaman asli
            crop_name_prediction = loaded_label_encoder.inverse_transform(numeric_prediction)

            st.markdown("---")
            st.subheader("✔️ Rekomendasi Untuk Anda:")
            st.success(f"**Tanaman yang paling cocok adalah: {crop_name_prediction[0]}**")
            st.balloons()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.error("Pastikan fungsi `predict_tree` dan `predict_random_forest` dari scratch sudah benar.")
    else:
        st.error("Model atau encoder tidak berhasil dimuat. Aplikasi tidak bisa melakukan prediksi.")

st.markdown("---")
st.markdown("Proyek Akhir Mata Kuliah AI | Model Klasifikasi Random Forest dari Scratch")
st.markdown(f"Waktu saat ini: {pd.Timestamp.now(tz='Asia/Jakarta').strftime('%A, %d %B %Y, %H:%M:%S %Z')}")