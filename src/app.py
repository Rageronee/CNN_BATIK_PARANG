
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

st.set_page_config(page_title="Batik Parang Classifier", layout="wide")

# Determine the directory where app.py is located
# For Hugging Face, app.py usually sits in the root.
# If running locally from root via `streamlit run src/app.py`, we need to handle paths carefully.

# Helper to find resources relative to current working directory or script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If app.py is inside src/, but models are in root (../models), we check parent.
# However, for Deployment, I recommend moving app.py to ROOT.
# This logic tries to find 'models' in current dir or parent dir.
if os.path.exists(os.path.join(BASE_DIR, 'models')):
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    PARANG_2_DIR = os.path.join(BASE_DIR, 'parang_2')
elif os.path.exists(os.path.join(os.path.dirname(BASE_DIR), 'models')):
    MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')
    PARANG_2_DIR = os.path.join(os.path.dirname(BASE_DIR), 'parang_2')
else:
    # Fallback to current working directory (useful for HF if structure is flat)
    MODEL_DIR = os.path.join(os.getcwd(), 'models')
    PARANG_2_DIR = os.path.join(os.getcwd(), 'parang_2')

# Define Class Labels (Hardcoded for now based on known structure)
CLASS_LABELS = {0: 'Non-Parang Rusak', 1: 'Parang Rusak'} # Adjust based on data_loader indices

@st.cache_resource
def load_models():
    models = {}
    resnet_path = os.path.join(MODEL_DIR, 'resnet50_best.h5')
    mobilenet_path = os.path.join(MODEL_DIR, 'mobilenetv2_best.h5')

    missing_models = []
    if not os.path.exists(resnet_path): missing_models.append('ResNet-50')
    if not os.path.exists(mobilenet_path): missing_models.append('MobileNetV2')

    if missing_models:
        st.warning(f"⚠️ Model berikut tidak ditemukan: {', '.join(missing_models)}. Pastikan Anda sudah meletakkan file model (.h5) di folder `models/`.")
        return models

    try:
        models['resnet50'] = tf.keras.models.load_model(resnet_path)
        models['mobilenetv2'] = tf.keras.models.load_model(mobilenet_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

st.title("🏯 Analisis Komparatif Klasifikasi Batik Parang")
st.markdown("""
Aplikasi ini membandingkan performa **ResNet-50** dan **MobileNetV2** dalam mengklasifikasikan motif Batik Parang.
Dataset uji 'Wild' diambil dari folder `parang_2`.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Citra")
    option = st.radio("Pilih Sumber Citra:", ("Upload Sendiri", "Pilih dari parang_2"))
    
    image_file = None
    if option == "Upload Sendiri":
        image_file = st.file_uploader("Upload Gambar JPG/PNG", type=['jpg', 'jpeg', 'png'])
    else:
        if os.path.exists(PARANG_2_DIR):
            files = [f for f in os.listdir(PARANG_2_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                st.warning("Folder `parang_2/` ditemukan, tetapi tidak ada gambar di dalamnya. Silakan tambahkan gambar `.jpg` atau `.png`.")
            else:
                selected_file = st.selectbox("Pilih Gambar:", files)
                if selected_file:
                    image_file = os.path.join(PARANG_2_DIR, selected_file)
        else:
            st.warning("Folder `parang_2/` tidak ditemukan. Silakan buat folder tersebut dan tambahkan gambar uji.")

    if image_file and models:
        img = Image.open(image_file)
        st.image(img, caption="Citra Input", use_container_width=True)
        
        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Klasifikasi"):
            st.divider()
            
            # Prediction Loop
            results = []
            for name, model in models.items():
                start = time.time()
                pred = model.predict(img_array)
                end = time.time()
                
                class_idx = np.argmax(pred)
                confidence = np.max(pred) * 100
                latency = (end - start) * 1000
                
                results.append({
                    'Model': name,
                    'Prediction': CLASS_LABELS.get(class_idx, "Unknown"),
                    'Confidence': confidence,
                    'Latency': latency
                })

            # Display Results
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🤖 MobileNetV2")
                res_mobile = next(r for r in results if r['Model'] == 'mobilenetv2')
                st.metric("Prediksi", res_mobile['Prediction'])
                st.progress(int(res_mobile['Confidence']))
                st.caption(f"Confidence: {res_mobile['Confidence']:.2f}% | Latency: {res_mobile['Latency']:.2f} ms")

            with c2:
                st.subheader("🧠 ResNet-50")
                res_resnet = next(r for r in results if r['Model'] == 'resnet50')
                st.metric("Prediksi", res_resnet['Prediction'])
                st.progress(int(res_resnet['Confidence']))
                st.caption(f"Confidence: {res_resnet['Confidence']:.2f}% | Latency: {res_resnet['Latency']:.2f} ms")
