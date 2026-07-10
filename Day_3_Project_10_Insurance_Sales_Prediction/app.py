import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Sales Prediction",
    page_icon="🏠",
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
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .buy-success {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .buy-warning {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
<div class="header-box">
    <h2>🏠 Project 10: Insurance Sales Prediction</h2>
    <p>Predict whether a person will buy life insurance based on their age using a trained Logistic Regression classification model.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# Load Dataset
# -----------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "insurance_data.csv")

if not os.path.exists(data_path):
    st.error(f"Error: {data_path} not found. Please ensure insurance_data.csv is placed in the project folder.")
else:
    df = pd.read_csv(data_path)
    
    # Left column for data and configuration, Right column for prediction and charts
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.subheader("📊 Insurance Dataset")
        st.dataframe(df, height=300)
        
        st.subheader("⚙️ Model Configuration")
        test_size = st.slider("Test Split Size (%)", 10, 50, 20, 5) / 100.0
        
        # -----------------------------------
        # Train Model
        # -----------------------------------
        # Separate features and target
        X = df[['age']]
        y = df['bought_insurance']
        
        # Split data with fixed random state to maintain model stability across run reruns
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train) * 100
        test_acc = model.score(X_test, y_test) * 100
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>💡 Model Training Accuracies</h4>
            <p style='font-size:1.1rem;'><b>Training Accuracy:</b> {train_acc:.1f}%</p>
            <p style='font-size:1.1rem;'><b>Testing Accuracy:</b> {test_acc:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🔮 Predict Insurance Sales")
        
        # User input for age
        age = st.slider("Select Person's Age:", min_value=10, max_value=70, value=30, step=1)
        
        st.write("") # Spacer
        
        # Prediction
        prediction = model.predict([[age]])[0]
        prob_buy = model.predict_proba([[age]])[0][1] * 100
        
        if prediction == 1:
            st.markdown(f"""
            <div class="buy-success">
                🎉 Prediction: This person is likely to BUY the insurance!<br>
                <span style='font-size:1rem; font-weight:normal;'>Buying Probability: {prob_buy:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="buy-warning">
                ❌ Prediction: This person is UNLIKELY to buy the insurance.<br>
                <span style='font-size:1rem; font-weight:normal;'>Buying Probability: {prob_buy:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") # Spacer
        
        # Sigmoid curve visualization
        st.subheader("📈 Logistic Sigmoid Curve Fit")
        
        # Generate prediction sequence for curve
        age_seq = np.linspace(10, 75, 300).reshape(-1, 1)
        prob_seq = model.predict_proba(age_seq)[:, 1]
        
        # Build Plotly chart
        fig = go.Figure()
        
        # Actual Data Points
        fig.add_trace(go.Scatter(
            x=df['age'],
            y=df['bought_insurance'],
            mode='markers',
            name='Actual Customers',
            marker=dict(
                size=10,
                color=df['bought_insurance'],
                colorscale=[[0, '#ef4444'], [1, '#10b981']],
                line=dict(color='black', width=1)
            ),
            hovertemplate='Age: %{x}<br>Bought: %{y}<extra></extra>'
        ))
        
        # Sigmoid Fit Curve
        fig.add_trace(go.Scatter(
            x=age_seq.flatten(),
            y=prob_seq,
            mode='lines',
            name='Logistic Fit S-Curve',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # User Selected Age Point
        fig.add_trace(go.Scatter(
            x=[age],
            y=[prob_buy / 100.0],
            mode='markers',
            name='Current Input Person',
            marker=dict(
                size=16,
                color='#f59e0b',
                symbol='star',
                line=dict(color='black', width=1.5)
            ),
            hovertemplate='Input Age: %{x}<br>Buying Probability: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Age (Years)",
            yaxis_title="Buying Probability",
            yaxis=dict(tickformat='.0%'),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
        fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("---")
        st.caption(f"**Model Statistics**: Coefficient = `{model.coef_[0][0]:.4f}` | Intercept = `{model.intercept_[0]:.4f}`")
