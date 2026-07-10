import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Page Config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .cat-result {
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #90caf9;
    }
    .dog-result {
        background-color: #efebe9;
        color: #4e342e;
        border: 1px solid #bcaaa4;
    }
</style>
<div class="header-box">
    <h2>🐱 Project 5: Cat vs Dog Image Classifier</h2>
    <p>Upload an image to predict whether it is a Cat or a Dog using a Transfer Learning Convolutional Neural Network (CNN).</p>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_cat_dog_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = base_dir
    while root_dir:
        if os.path.exists(os.path.join(root_dir, "models")):
            break
        parent = os.path.dirname(root_dir)
        if parent == root_dir:
            break
        root_dir = parent
    model_path = os.path.join(root_dir, "models", "cat_dog_model.h5")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

model = load_cat_dog_model()

if model is None:
    st.error("❌ Model file not found. Please verify `cat_dog_model.h5` exists in the `models/` directory.")
else:
    IMG_SIZE = 64

    # Upload Image
    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load and display image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_container_width=True)
        
        # Preprocess image
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to float array and preprocess matching MobileNetV2
        img_array = np.array(img_resized, dtype=np.float32)
        img_preprocessed = preprocess_input(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Prediction
        with st.spinner("Analyzing image..."):
            pred_prob = float(model.predict(img_batch)[0][0])
            
            # Squeeze output to ensure it stays in bounds [0, 1]
            pred_prob = np.clip(pred_prob, 0.0, 1.0)
            
            dog_prob = pred_prob * 100
            cat_prob = (1.0 - pred_prob) * 100
            
            # Output display
            if pred_prob <= 0.5:
                confidence = cat_prob
                st.markdown(f'<div class="result-text cat-result">🐱 Predicted: Cat ({confidence:.1f}% confidence)</div>', unsafe_allow_html=True)
            else:
                confidence = dog_prob
                st.markdown(f'<div class="result-text dog-result">🐶 Predicted: Dog ({confidence:.1f}% confidence)</div>', unsafe_allow_html=True)
                
            st.write("---")
            st.subheader("Probability Distribution:")
            col1, col2 = st.columns(2)
            col1.metric("🐱 Cat Probability", f"{cat_prob:.1f}%")
            col2.metric("🐶 Dog Probability", f"{dog_prob:.1f}%")
            
            st.progress(cat_prob / 100.0, text=f"Cat Probability: {cat_prob:.1f}%")
            st.progress(dog_prob / 100.0, text=f"Dog Probability: {dog_prob:.1f}%")
