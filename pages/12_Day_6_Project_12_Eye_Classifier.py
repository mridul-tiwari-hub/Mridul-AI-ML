import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Male/Female Eye Classifier",
    page_icon="👁️",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .status-female {
        background-color: #fdf2f8;
        color: #9d174d;
        border: 1px solid #fbcfe8;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    .status-male {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
</style>
<div class="header-box">
    <h2>👁️ Project 12: Male/Female Eye Classifier</h2>
    <p>Upload a cropped close-up image of a human eye to predict whether it is a male or female eye using a custom-trained Convolutional Neural Network (CNN) model.</p>
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
model_path = os.path.join(root_dir, "models", "eyes_cnn_model.h5")

@st.cache_resource
def load_eyes_model():
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

model = load_eyes_model()

if model is None:
    st.error(f"❌ Model file not found at `{model_path}`. Please verify model weights exist in the `models/` directory.")
else:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📤 Upload Eye Image")
        uploaded_file = st.file_uploader(
            "Upload a cropped eye image (JPG, JPEG, PNG)...",
            type=["jpg", "jpeg", "png"]
        )
        
        st.write("---")
        st.subheader("💡 Testing & Sample Data")
        st.write("If you don't have a cropped eye image, you can download these sample images from the training dataset:")
        
        col_f, col_m = st.columns(2)
        
        with col_f:
            f_sample_path = os.path.join(root_dir, "Day_6_Project_12_Eye_Classifier", "samples", "female_eye_sample.jpg")
            if os.path.exists(f_sample_path):
                with open(f_sample_path, "rb") as f:
                    st.download_button(
                        label="👩 Download Female Eye",
                        data=f.read(),
                        file_name="female_eye_sample.jpg",
                        mime="image/jpeg"
                    )
            else:
                st.caption("Female eye sample not found.")
                
        with col_m:
            m_sample_path = os.path.join(root_dir, "Day_6_Project_12_Eye_Classifier", "samples", "male_eye_sample.jpg")
            if os.path.exists(m_sample_path):
                with open(m_sample_path, "rb") as f:
                    st.download_button(
                        label="👨 Download Male Eye",
                        data=f.read(),
                        file_name="male_eye_sample.jpg",
                        mime="image/jpeg"
                    )
            else:
                st.caption("Male eye sample not found.")
            
    with col2:
        st.subheader("🔍 Prediction Results")
        if uploaded_file is not None:
            # Display image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Eye Image", width=250)
            
            # Preprocess image
            img_resized = img.resize((64, 64))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            with st.spinner("Analyzing image..."):
                prediction = model.predict(img_array)
                probability = float(prediction[0][0])
                
                # Clip values to ensure progress bar boundary safety
                probability = np.clip(probability, 0.0, 1.0)
                
                st.write("") # Spacer
                if probability >= 0.5:
                    confidence = probability * 100
                    st.markdown(f"""
                    <div class="status-male">
                        👨 Predicted Class: MALE EYE<br>
                        <span style='font-size:1.1rem; font-weight:normal;'>Confidence Score: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1.0 - probability) * 100
                    st.markdown(f"""
                    <div class="status-female">
                        👩 Predicted Class: FEMALE EYE<br>
                        <span style='font-size:1.1rem; font-weight:normal;'>Confidence Score: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                st.write("**Confidence Level Indicator**")
                st.progress(float(confidence / 100.0))
        else:
            st.info("Upload a cropped eye scan in the left panel to begin classification.")
            
        st.write("---")
        with st.expander("🛠️ Model Architecture Details"):
            st.markdown("""
            This model is a custom **Convolutional Neural Network (CNN)** trained on cropped eye images:
            - **Input shape**: `64x64` RGB image
            - **Layers**:
              - 2D Convolution (32 filters, 3x3 kernel, Relu) + Max Pooling (2x2)
              - 2D Convolution (64 filters, 3x3 kernel, Relu) + Max Pooling (2x2)
              - 2D Convolution (128 filters, 3x3 kernel, Relu) + Max Pooling (2x2)
              - Flatten + Dense hidden layer (64 units, Relu)
              - Dropout layer (rate 0.5) to prevent overfitting
              - Dense sigmoid layer (1 unit) to classify **Female (0)** vs **Male (1)**.
            """)
