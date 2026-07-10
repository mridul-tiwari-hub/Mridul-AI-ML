import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detector",
    page_icon="🩻",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .status-covid {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    .status-normal {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
<div class="header-box">
    <h2>🩻 Project 11: COVID-19 Chest X-Ray Detector</h2>
    <p>Upload a patient's chest X-Ray image to classify it as either normal or showing signs of COVID-19 infection using a custom-trained Convolutional Neural Network (CNN).</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# Load Pre-trained Model
# -----------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = base_dir
while root_dir:
    if os.path.exists(os.path.join(root_dir, "models")):
        break
    parent = os.path.dirname(root_dir)
    if parent == root_dir:
        break
    root_dir = parent
model_path = os.path.join(root_dir, "models", "covid_model.keras")

@st.cache_resource
def load_covid_model():
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

model = load_covid_model()

if model is None:
    st.error(f"❌ Model file not found at `{model_path}`. Please verify model weights exist in the `models/` directory.")
else:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📤 Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Choose a chest X-Ray image (JPG, JPEG, PNG)...",
            type=["jpg", "jpeg", "png"]
        )
        
        st.write("---")
        st.subheader("💡 Testing & Sample Data")
        st.write("If you don't have an X-Ray image on hand, you can download this standard clinical Normal Chest X-Ray sample:")
        
        sample_img_path = os.path.join(root_dir, "Day_6_Project_11_Covid19_Detector", "normal_sample.jpg")
        if os.path.exists(sample_img_path):
            with open(sample_img_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Normal X-Ray Sample",
                    data=f.read(),
                    file_name="normal_xray_sample.jpg",
                    mime="image/jpeg"
                )
        else:
            st.caption("Normal sample image not found in folder.")
            
    with col2:
        st.subheader("🔍 Prediction Results")
        if uploaded_file is not None:
            # Display image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Chest X-Ray", use_container_width=True)
            
            # Preprocess image
            img_resized = img.resize((299, 299))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            with st.spinner("Analyzing scan..."):
                prediction = model.predict(img_array)
                probability = float(prediction[0][0])
                
                # Squeeze outputs and clip progress boundaries
                probability = np.clip(probability, 0.0, 1.0)
                
                st.write("") # Spacer
                if probability > 0.5:
                    confidence = probability * 100
                    st.markdown(f"""
                    <div class="status-covid">
                        🦠 Predicted Status: COVID-19 Positive<br>
                        <span style='font-size:1.1rem; font-weight:normal;'>Probability of abnormality: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1.0 - probability) * 100
                    st.markdown(f"""
                    <div class="status-normal">
                        ✅ Predicted Status: NORMAL<br>
                        <span style='font-size:1.1rem; font-weight:normal;'>Probability of normality: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                st.write("**Confidence Level Indicator**")
                st.progress(float(confidence / 100.0))
        else:
            st.info("Upload a chest X-Ray image in the left panel to begin classification.")
            
        st.write("---")
        with st.expander("🛠️ Model Architecture Details"):
            st.markdown("""
            This model is a custom **Convolutional Neural Network (CNN)** containing:
            - **4 Convolutional Layers** (with Relu activation, sizes 32, 32, 64, and 128)
            - **4 Max Pooling Layers**
            - **Flatten layer** to convert 2D feature maps to 1D
            - **Dense hidden layer** with 128 units
            - **Dense classification layer** with Sigmoid activation outputting binary probability (COVID vs Normal).
            """)
