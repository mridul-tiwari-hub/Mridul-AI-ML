import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Page Config
st.set_page_config(
    page_title="Iris Flower Classifier - KNN",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .result-text {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .setosa-result {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    .versicolor-result {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffe0b2;
    }
    .virginica-result {
        background-color: #ede7f6;
        color: #4527a0;
        border: 1px solid #d1c4e9;
    }
</style>
<div class="header-box">
    <h2>🌸 Project 8: Iris Flower Classifier (KNN Deployment)</h2>
    <p>Predict the iris flower species based on physical measurements using a trained K-Nearest Neighbors (KNN) model.</p>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_knn_model():
    model_path = os.path.join("models", "knn_iris_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found.")
    return joblib.load(model_path)

try:
    model = load_knn_model()
except Exception as e:
    st.error(f"Error loading KNN model: {e}")
    model = None

# Load dataset for reference visualization
iris = load_iris()
df_ref = pd.DataFrame(iris.data, columns=iris.feature_names)
df_ref['species'] = iris.target
species_names = ['Setosa', 'Versicolor', 'Virginica']

st.write("Enter the sepal and petal dimensions below to predict the flower species:")

# Input layout
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", float(df_ref.iloc[:,0].min()), float(df_ref.iloc[:,0].max()), 5.4, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", float(df_ref.iloc[:,1].min()), float(df_ref.iloc[:,1].max()), 3.4, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", float(df_ref.iloc[:,2].min()), float(df_ref.iloc[:,2].max()), 1.5, 0.1)
    petal_width = st.slider("Petal Width (cm)", float(df_ref.iloc[:,3].min()), float(df_ref.iloc[:,3].max()), 0.4, 0.1)

if model is not None:
    # Run Prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Class style mapping
    styles = ["setosa-result", "versicolor-result", "virginica-result"]
    predicted_species = species_names[prediction]
    conf_style = styles[prediction]
    confidence = probabilities[prediction] * 100
    
    st.markdown(f'<div class="result-text {conf_style}">🌸 Predicted Species: {predicted_species} ({confidence:.1f}% probability)</div>', unsafe_allow_html=True)
    
    # Probabilities breakdown
    st.subheader("Species Probabilities:")
    cols = st.columns(3)
    for idx, col in enumerate(cols):
        prob = probabilities[idx] * 100
        col.metric(f"🌸 {species_names[idx]}", f"{prob:.1f}%")
        
    st.write("---")
    
    # Interactive scatter plot
    st.subheader("Your Input Position relative to training set:")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot reference data
    colors = ['#2e7d32', '#ef6c00', '#4527a0']
    for idx, spec in enumerate(species_names):
        spec_data = df_ref[df_ref['species'] == idx]
        ax.scatter(spec_data['petal length (cm)'], spec_data['petal width (cm)'], label=spec, color=colors[idx], alpha=0.6)
        
    # Plot user point
    ax.scatter([petal_length], [petal_width], color='red', marker='*', s=250, label='Your Input', edgecolors='black')
    
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title("Petal Length vs Width Clustering")
    ax.legend()
    st.pyplot(fig)
