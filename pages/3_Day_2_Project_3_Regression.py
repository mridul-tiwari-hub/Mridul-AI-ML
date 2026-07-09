import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(
    page_title="Canada Income Predictor",
    page_icon="🇨🇦",
    layout="wide"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🇨🇦 Project 3: Canada Per Capita Income Prediction</h2>
    <p>Predict Canada's per capita income in US$ using a simple Linear Regression model trained on historical data from 1970 to 2016.</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    file_path = os.path.join("data", "canada_per_capita_income.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path {file_path} not found.")
    df = pd.read_csv(file_path)
    # Ensure column naming is clean
    df.columns = ['year', 'income']
    return df

try:
    df = load_data()
    
    # Train Linear Regression model
    X = df[['year']]
    y = df['income']
    model = LinearRegression()
    model.fit(X, y)
    
    coef = model.coef_[0]
    intercept = model.intercept_
    
    # Input panel
    st.subheader("🔮 Predict Income")
    predict_year = st.slider("Select Target Year to Predict", 1970, 2040, 2020, 1)
    
    # Run prediction
    pred_income = model.predict(np.array([[predict_year]]))[0]
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Predicted Income for {predict_year}", f"${pred_income:,.2f} USD")
    col2.metric("Model Slope (m)", f"{coef:.2f}")
    col3.metric("Model Intercept (b)", f"{intercept:,.2f}")
    
    st.write("---")
    
    # Visualisation
    st.write("### 📈 Linear Regression Trend Line vs Actual Data")
    
    # Generate regression line points
    years_range = np.linspace(df['year'].min(), 2040, 100).reshape(-1, 1)
    predicted_line = model.predict(years_range)
    
    fig = go.Figure()
    
    # Add actual data points
    fig.add_trace(go.Scatter(
        x=df['year'], 
        y=df['income'], 
        mode='markers',
        name='Actual Income Data',
        marker=dict(color='#3b82f6', size=8)
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=years_range.flatten(), 
        y=predicted_line, 
        mode='lines',
        name='Linear Regression Line',
        line=dict(color='#ef4444', width=2)
    ))
    
    # Add predicted point
    fig.add_trace(go.Scatter(
        x=[predict_year], 
        y=[pred_income], 
        mode='markers',
        name='Your Prediction Point',
        marker=dict(color='#f59e0b', size=15, symbol='star', line=dict(color='black', width=1.5))
    ))
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Per Capita Income (US$)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Formula representation
    st.write("### 📝 Regression Equation")
    st.info(f"**Income = ({coef:.4f} * Year) + ({intercept:.4f})**")
    
except Exception as e:
    st.error(f"Error executing regression logic: {e}")
