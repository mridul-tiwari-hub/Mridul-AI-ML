import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Page Config
st.set_page_config(
    page_title="Gender Recognition App",
    page_icon="👤",
    layout="centered"
)

# Custom styling for premium interface
st.markdown("""
<style>
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .female-result {
        background-color: #fce4ec;
        color: #c2185b;
        border: 1px solid #f8bbd0;
    }
    .male-result {
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #90caf9;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_cnn_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gender_cnn_model.h5")
    return tf.keras.models.load_model(model_path)

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Header
st.title("👤 Gender Recognition CNN Application")
st.write("Upload an image of a face to predict whether it is Male or Female using a Convolutional Neural Network (CNN).")

# Image Uploader
uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Read & Display Image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess Image to match model's expected shape: (1, 150, 150, 3)
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Prediction
    pred = model.predict(img_batch)[0][0]
    
    # Probabilities
    # Alphabetical order: Class 0 = Female, Class 1 = Male
    male_prob = pred * 100
    female_prob = (1 - pred) * 100
    
    if pred >= 0.5:
        st.markdown(f'<div class="result-text male-result">👨 Predicted: Male ({male_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-text female-result">👩 Predicted: Female ({female_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
        
    st.write("---")
    st.subheader("Probability Distribution:")
    col1, col2 = st.columns(2)
    col1.metric("👩 Female Probability", f"{female_prob:.1f}%")
    col2.metric("👨 Male Probability", f"{male_prob:.1f}%")
    
    st.progress(female_prob / 100.0, text=f"Female Probability: {female_prob:.1f}%")
    st.progress(male_prob / 100.0, text=f"Male Probability: {male_prob:.1f}%")
