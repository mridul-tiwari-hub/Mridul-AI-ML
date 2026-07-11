import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Life Insurance Purchase Prediction",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Life Insurance Purchase Prediction")
st.markdown("Predict whether a person will buy life insurance based on their age using Logistic Regression.")

# Load Dataset
df = pd.read_csv("insurance_data.csv")

st.header("Dataset Overview")
st.dataframe(df, use_container_width=True)

# Train Model
X = df[["age"]]
y = df["bought_insurance"]
model = LogisticRegression(solver="lbfgs")
model.fit(X, y)

# Input for Age
st.header("Enter Customer Details")
age_input = st.number_input("Age of the Person", min_value=1, max_value=120, value=35, step=1)

# Predict button
if st.button("Predict Insurance Status", use_container_width=True):
    prediction = model.predict([[age_input]])[0]
    proba = model.predict_proba([[age_input]])[0]
    
    st.write("")
    if prediction == 1:
        st.success(f"Prediction: This person WILL buy insurance.", icon="✅")
    else:
        st.warning(f"Prediction: This person WILL NOT buy insurance.", icon="⚠️")
    st.info(f"Probability of buying: {proba[1]*100:.2f}%", icon="📊")

# Model Parameters
st.header("Model Parameters")
st.markdown(f"**Coefficient (m):** `{model.coef_[0][0]}`")
st.markdown(f"**Intercept (b):** `{model.intercept_[0]}`")