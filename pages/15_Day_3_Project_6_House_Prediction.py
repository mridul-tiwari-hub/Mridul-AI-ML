import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🏠 Day 3: House Price Prediction</h2>
    <p>Predict house price using Linear Regression trained on area size.</p>
</div>
""", unsafe_allow_html=True)

# Load Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "..", "Day_3_Project_6_House_Prediction", "houseprice.csv")

if not os.path.exists(dataset_path):
    st.error(f"⚠️ Dataset file `houseprice.csv` not found at {dataset_path}.")
    st.stop()

df = pd.read_csv(dataset_path)

st.subheader("📋 Dataset Overview")
st.dataframe(df, use_container_width=True)

# Train Model
X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# User Input
st.subheader("🔑 Predict New House Price")
area = st.number_input(
    "Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=3300,
    step=100
)

# Prediction
if st.button("Predict Price", use_container_width=True):
    prediction = model.predict([[area]])
    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

# Model Information
st.subheader("ℹ️ Model Details")
col1, col2 = st.columns(2)
col1.metric("Coefficient (Slope)", f"{model.coef_[0]:,.4f}")
col2.metric("Intercept", f"{model.intercept_:,.4f}")
