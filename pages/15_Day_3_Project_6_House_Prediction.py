import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Life Insurance Purchase Prediction",
    page_icon="🛡️",
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
</style>
<div class="header-box">
    <h2>🛡️ Life Insurance Purchase Prediction</h2>
    <p>Predict whether a person will buy life insurance based on their age using a Logistic Regression model.</p>
</div>
""", unsafe_allow_html=True)

# Load Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "..", "Day_3_Project_6_House_Prediction", "insurance_data.csv")

if not os.path.exists(dataset_path):
    st.error(f"⚠️ Dataset file `insurance_data.csv` not found at {dataset_path}.")
    st.stop()

df = pd.read_csv(dataset_path)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df, use_container_width=True, height=300)

    # Train Model
    X = df[["age"]]
    y = df["bought_insurance"]
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)

    st.subheader("ℹ️ Model Parameters")
    st.markdown(f"**Coefficient (m):** `{model.coef_[0][0]:.4f}`")
    st.markdown(f"**Intercept (b):** `{model.intercept_[0]:.4f}`")

with col2:
    st.subheader("🔑 Predict Insurance Status")
    age_input = st.number_input(
        "Age of the Person",
        min_value=1,
        max_value=120,
        value=35,
        step=1
    )

    if st.button("Predict Purchase Likelihood", use_container_width=True):
        prediction = model.predict([[age_input]])[0]
        proba = model.predict_proba([[age_input]])[0]

        st.write("")
        if prediction == 1:
            st.success(f"**Prediction:** This person **WILL** buy insurance.", icon="✅")
        else:
            st.warning(f"**Prediction:** This person **WILL NOT** buy insurance.", icon="⚠️")
        
        st.info(f"**Probability of buying:** {proba[1]*100:.2f}%", icon="📊")
