import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import os

# Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🏡 Project 6: House Price Predictor</h2>
    <p>Estimate property prices using a Multiple Linear Regression model trained on Area, Bedrooms, and Age.</p>
</div>
""", unsafe_allow_html=True)

# Load Dataset
dataset_path = "houseprice.csv"
if not os.path.exists(dataset_path):
    st.error("⚠️ Dataset file `houseprice.csv` not found in current directory.")
    st.stop()

df = pd.read_csv(dataset_path)

# Train Multiple Linear Regression
X = df[["area", "bedrooms", "age"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
df_with_pred = df.copy()
df_with_pred["predicted_price"] = y_pred

# Layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df, use_container_width=True, height=280)
    
    st.subheader("⚙️ Model Parameters")
    st.markdown(f"**Area Coefficient (β₁):** `{model.coef_[0]:.2f}`")
    st.markdown(f"**Bedrooms Coefficient (β₂):** `{model.coef_[1]:.2f}`")
    st.markdown(f"**Age Coefficient (β₃):** `{model.coef_[2]:.2f}`")
    st.markdown(f"**Intercept (α):** `{model.intercept_:.2f}`")

with col2:
    st.subheader("🔮 Predict Price")
    area_input = st.number_input("Area (Square Feet)", min_value=500, max_value=10000, value=1800, step=50)
    bedrooms_input = st.slider("Number of Bedrooms", min_value=1, max_value=8, value=3)
    age_input = st.number_input("Age of the House (Years)", min_value=0, max_value=100, value=10)
    
    if st.button("Calculate Predicted Price", use_container_width=True):
        predicted_val = model.predict([[area_input, bedrooms_input, age_input]])[0]
        
        st.write("")
        st.success(f"**Estimated Price:** ${predicted_val:,.2f} USD", icon="✅")
        
        # Display equation application
        st.info(
            f"Calculation: ({model.coef_[0]:.2f} * {area_input}) + "
            f"({model.coef_[1]:.2f} * {bedrooms_input}) + "
            f"({model.coef_[2]:.2f} * {age_input}) + "
            f"({model.intercept_:.2f}) = **${predicted_val:,.2f}**",
            icon="ℹ️"
        )