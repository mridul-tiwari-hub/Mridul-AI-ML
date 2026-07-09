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
    <p>Upload an image to predict whether it is a Cat or a Dog. The image is flattened and run through a trained classification model.</p>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) == "pages" else current_dir
    model_path = os.path.join(root_dir, "models", "cat_dog_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found.")
    return joblib.load(model_path)

try:
    model = load_model()
    
    IMG_SIZE = 64
    classes = ['Cat', 'Dog']

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

except Exception as e:
    st.error(f"Error loading or running the Cat vs Dog model: {e}")
