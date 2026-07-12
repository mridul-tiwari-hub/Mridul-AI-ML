import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# Load Model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cat_dog_model.pkl")
model = joblib.load(model_path)

IMG_SIZE = 64
classes = ['Cat', 'Dog']

# Title & Info
st.title("🐱 Cat vs Dog Image Classifier")
st.write("Upload an image to predict whether it is a Cat or Dog.")

# Custom CSS for styling
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
""", unsafe_allow_html=True)

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
    
    # Convert RGB to BGR (to match cv2 format used during training)
    img_bgr = np.array(img_resized)[:, :, ::-1]
    
    # Flatten and normalize
    features = img_bgr.flatten().reshape(1, -1) / 255.0
    
    # Prediction
    pred_class = model.predict(features)[0]
    pred_prob = model.predict_proba(features)[0]
    
    cat_prob = pred_prob[0] * 100
    dog_prob = pred_prob[1] * 100
    
    # Output display
    if pred_class == 0:
        st.markdown(f'<div class="result-text cat-result">🐱 Predicted: Cat ({cat_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-text dog-result">🐶 Predicted: Dog ({dog_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
        
    st.write("---")
    st.subheader("Probability Distribution:")
    col1, col2 = st.columns(2)
    col1.metric("🐱 Cat Probability", f"{cat_prob:.1f}%")
    col2.metric("🐶 Dog Probability", f"{dog_prob:.1f}%")
    
    st.progress(cat_prob / 100.0, text=f"Cat Probability: {cat_prob:.1f}%")
    st.progress(dog_prob / 100.0, text=f"Dog Probability: {dog_prob:.1f}%")
