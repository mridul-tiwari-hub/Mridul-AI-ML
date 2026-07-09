import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import os

# Page Config
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="💼",
    layout="wide"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .status-text {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .leave-status {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fee2e2;
    }
    .stay-status {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #dcfce7;
    }
</style>
<div class="header-box">
    <h2>💼 Project 4: HR Employee Retention Prediction</h2>
    <p>Analyze employee turnover factors and predict leaving probabilities using a Logistic Regression classifier trained on real HR data.</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "HR_comma_sep.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path {file_path} not found.")
    df = pd.read_csv(file_path)
    return df

try:
    df = load_data()
    
    # Train model
    # Prepare features: satisfaction_level, average_montly_hours, promotion_last_5years, salary
    subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
    salary_dummies = pd.get_dummies(subdf['salary'], prefix="salary", dtype=float)
    df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
    df_with_dummies.drop('salary', axis='columns', inplace=True)
    
    # Features (X) & Target (y)
    X = df_with_dummies.drop('salary_high', axis='columns')
    y = df['left']
    
    # Train test split for basic scoring
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Split UI
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.write("### 🏢 Retention Predictive Model")
        st.write("Adjust the employee metrics below to test their retention probability:")
        
        satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.05)
        avg_hours = st.slider("Average Monthly Hours", 90, 320, 200, 5)
        promotion = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])
        salary = st.selectbox("Salary Tier", ["Low", "Medium", "High"])
        
        # Prepare user input row
        # Feature columns: satisfaction_level, average_montly_hours, promotion_last_5years, salary_low, salary_medium
        promo_val = 1.0 if promotion == "Yes" else 0.0
        sal_low_val = 1.0 if salary == "Low" else 0.0
        sal_med_val = 1.0 if salary == "Medium" else 0.0
        
        user_input = np.array([[satisfaction, avg_hours, promo_val, sal_low_val, sal_med_val]])
        prediction = model.predict(user_input)[0]
        probabilities = model.predict_proba(user_input)[0]
        
        stay_prob = probabilities[0] * 100
        leave_prob = probabilities[1] * 100
        
        if prediction == 1:
            st.markdown(f'<div class="status-text leave-status">⚠️ Predicted: Likely to Leave ({leave_prob:.1f}% probability)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-text stay-status">✅ Predicted: Likely to Stay ({stay_prob:.1f}% probability)</div>', unsafe_allow_html=True)
            
        st.write("#### Retention Probability Breakdown:")
        st.progress(stay_prob / 100.0, text=f"Stay Probability: {stay_prob:.1f}%")
        st.progress(leave_prob / 100.0, text=f"Leave Probability: {leave_prob:.1f}%")
        st.caption(f"Model test-set accuracy: {accuracy * 100:.2f}%")
        
    with col_right:
        st.write("### 📊 Exploratory Insights from Dataset")
        
        insight_type = st.radio("Select Analysis Factor:", [
            "Impact of Salary Tier on Leaving",
            "Impact of Promotion on Leaving",
            "Satisfaction Level vs Retention"
        ])
        
        if insight_type == "Impact of Salary Tier on Leaving":
            salary_left = pd.crosstab(df['salary'], df['left'])
            salary_left.columns = ['Retained', 'Left']
            salary_left = salary_left.reset_index()
            fig = px.bar(
                salary_left, 
                x='salary', 
                y=['Retained', 'Left'], 
                title="Employee Turnover by Salary Level",
                barmode='group',
                color_discrete_map={'Retained': '#10b981', 'Left': '#ef4444'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif insight_type == "Impact of Promotion on Leaving":
            promo_left = pd.crosstab(df['promotion_last_5years'], df['left'])
            promo_left.columns = ['Retained', 'Left']
            promo_left.index = ['No Promotion', 'Promoted']
            promo_left = promo_left.reset_index()
            fig = px.bar(
                promo_left, 
                x='index', 
                y=['Retained', 'Left'], 
                title="Turnover Rate based on Promotion status",
                barmode='group',
                color_discrete_map={'Retained': '#10b981', 'Left': '#ef4444'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            fig = px.histogram(
                df, 
                x="satisfaction_level", 
                color="left", 
                title="Satisfaction Level Distribution (0 = Retained, 1 = Left)",
                color_discrete_sequence=['#10b981', '#ef4444'],
                opacity=0.7,
                barmode='overlay',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error executing employee retention regression: {e}")
