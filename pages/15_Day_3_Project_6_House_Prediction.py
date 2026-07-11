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

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .header-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
</style>
<div class="header-box">
    <div class="header-title">🏡 Project 6: House Price Predictor</div>
    <div class="header-subtitle">Estimate residential property values using a Multiple Linear Regression model trained on Area, Bedrooms, and Age.</div>
</div>
""", unsafe_allow_html=True)

# Load Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "..", "Day_3_Project_6_House_Prediction", "houseprice.csv")

if not os.path.exists(dataset_path):
    st.error(f"⚠️ Dataset file `houseprice.csv` not found at {dataset_path}.")
    st.stop()

# Cache data loading
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(dataset_path)

# Train Multiple Linear Regression Model
X = df[["area", "bedrooms", "age"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# Predict for all data to show metrics
y_pred = model.predict(X)
df_with_pred = df.copy()
df_with_pred["predicted_price"] = y_pred

# Layout columns
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📊 Dataset Explorer")
    st.markdown("Here is the historical housing dataset used to train the regression model:")
    st.dataframe(df, use_container_width=True, height=280)
    
    st.subheader("⚙️ Model Parameters & Coefficients")
    st.markdown("The linear equation calculated by the model is:")
    
    # Mathematical expression HTML
    equation_html = f"""
    <div style="background-color: #f8fafc; padding: 1rem; border-left: 5px solid #3b82f6; border-radius: 6px; font-family: monospace; font-size: 0.95rem; margin-bottom: 1.5rem;">
        Price = ({model.coef_[0]:.2f} * Area) + ({model.coef_[1]:.2f} * Bedrooms) + ({model.coef_[2]:.2f} * Age) + ({model.intercept_:.2f})
    </div>
    """
    st.markdown(equation_html, unsafe_allow_html=True)
    
    # Parameter metrics
    m_cols = st.columns(4)
    m_cols[0].metric("Area Coeff (β₁)", f"{model.coef_[0]:.2f}")
    m_cols[1].metric("Bedrooms Coeff (β₂)", f"{model.coef_[1]:,.2f}")
    m_cols[2].metric("Age Coeff (β₃)", f"{model.coef_[2]:.2f}")
    m_cols[3].metric("Intercept (α)", f"{model.intercept_:,.0f}")

with col2:
    st.subheader("🔮 Predict Property Value")
    st.markdown("Adjust property attributes below to obtain a real-time price prediction:")
    
    # Property Input Fields
    area_input = st.number_input(
        "Lot Area (Square Feet)",
        min_value=500,
        max_value=10000,
        value=1800,
        step=50
    )
    
    bed_col, age_col = st.columns(2)
    with bed_col:
        bedrooms_input = st.slider(
            "Number of Bedrooms",
            min_value=1,
            max_value=8,
            value=3,
            step=1
        )
    with age_col:
        age_input = st.number_input(
            "Age of the House (Years)",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )
    
    st.markdown("---")
    
    # Calculate Prediction
    predicted_val = model.predict([[area_input, bedrooms_input, age_input]])[0]
    
    # Display Prediction Result card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8;">Estimated Market Value</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #10b981; margin: 0.5rem 0;">${predicted_val:,.2f} USD</div>
        <div style="font-size: 0.85rem; opacity: 0.7;">Based on Multiple Linear Regression analysis</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Visualizations Section
st.subheader("📈 Visualization & Fit Metrics")
viz_col1, viz_col2 = st.columns(2, gap="large")

with viz_col1:
    st.write("#### 🏠 Area vs Price (with predicted trend line)")
    
    # Create regression line plot
    fig_scatter = px.scatter(
        df, 
        x="area", 
        y="price", 
        color="age",
        size="bedrooms",
        labels={"area": "Area (Sq Ft)", "price": "Price ($)", "age": "Age (Years)", "bedrooms": "Bedrooms"},
        title="Housing Price Distribution by Area, Age & Bedrooms"
    )
    
    # Add predicted point
    fig_scatter.add_trace(go.Scatter(
        x=[area_input],
        y=[predicted_val],
        mode="markers",
        name="Your Predicted House",
        marker=dict(color="#10b981", size=18, symbol="star", line=dict(color="white", width=2))
    ))
    
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with viz_col2:
    st.write("#### 🎯 Model Fit: Actual vs. Predicted Prices")
    
    # Scatter plot of actual vs predicted prices
    fig_fit = px.scatter(
        df_with_pred,
        x="price",
        y="predicted_price",
        labels={"price": "Actual Price ($)", "predicted_price": "Predicted Price ($)"},
        title="Actual Price vs. Model Predicted Price"
    )
    
    # Add a 45-degree line representing perfect prediction
    min_val = min(df["price"].min(), df_with_pred["predicted_price"].min())
    max_val = max(df["price"].max(), df_with_pred["predicted_price"].max())
    fig_fit.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash", width=2),
        name="Perfect Fit Line"
    )
    
    fig_fit.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_fit, use_container_width=True)
