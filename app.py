import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Page Config
st.set_page_config(
    page_title="Employee Retention Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply some custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Employee Retention App")
st.sidebar.markdown("---")
st.sidebar.write("This application predicts whether an employee will stay in the company or leave based on working conditions like salary, satisfaction level, working hours, and experience.")
st.sidebar.markdown("### Model Used:")
st.sidebar.info("🧠 **Logistic Regression**")

# Main Title
st.title("Employee Retention Prediction App")

# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/aiplanethub/Datasets/refs/heads/master/HR_comma_sep.csv"
    data = pd.read_csv(url)
    return data

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading dataset from URL: {e}")
    st.stop()

# Overview section
st.header("Overview")
st.write(
    "This application predicts whether an employee will stay in the company or leave based on working conditions like salary, satisfaction level, working hours, and experience."
)

st.subheader("Objective")
st.markdown("""
* 📊 **Analyze employee behavior** to detect key churn drivers.
* 🔍 **Identify reasons for employee attrition** using data-driven insights.
* 🤖 **Build a machine learning model for prediction** of retention risks.
""")

st.markdown("---")

# Dataset Preview
st.header("Dataset Preview")
st.dataframe(df_raw, use_container_width=True)

# Dataset Information
st.header("Dataset Information")
col_info, col_clean = st.columns(2)

with col_info:
    st.subheader("Data Types & Counts")
    info_df = pd.DataFrame({
        "Column Name": df_raw.columns,
        "Non-Null Count": df_raw.notnull().sum().values,
        "Data Type": [str(t) for t in df_raw.dtypes.values]
    })
    st.table(info_df)

# Data Cleaning & Duplicates
df_clean = df_raw.drop_duplicates()
duplicate_count = len(df_raw) - len(df_clean)

with col_clean:
    st.subheader("Column Names & Data Cleaning")
    st.markdown(f"""
    * **Original Record Count:** `{len(df_raw)}`
    * **Duplicates Found & Removed:** `{duplicate_count}`
    * **Cleaned Record Count:** `{len(df_clean)}`
    """)
    st.success(f"🧹 Cleaned dataset successfully! Removed {duplicate_count} duplicate rows.")

st.markdown("---")

# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis (EDA)")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Salary vs Employee Retention")
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df_clean, x='salary', hue='left', ax=ax1, palette=['#1f77b4', '#ff7f0e'])
    ax1.set_xlabel("Salary Level", fontsize=10)
    ax1.set_ylabel("Number of Employees", fontsize=10)
    ax1.legend(["Stayed (0)", "Left (1)"], title="Attrition Status")
    fig1.tight_layout()
    st.pyplot(fig1)

with eda_col2:
    st.subheader("Department-wise Employee Retention")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=df_clean, x='Department', hue='left', ax=ax2, palette=['#1f77b4', '#ff7f0e'])
    ax2.set_xlabel("Department", fontsize=10)
    ax2.set_ylabel("Number of Employees", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    ax2.legend(["Stayed (0)", "Left (1)"], title="Attrition Status")
    fig2.tight_layout()
    st.pyplot(fig2)

st.markdown("---")

# Preprocessing & Model Building
# Categorical Encoding
df_encoded = pd.get_dummies(df_clean, columns=['Department', 'salary'], drop_first=True)
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cache model training
@st.cache_resource
def train_model(X_tr, y_tr):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_tr, y_tr)
    return lr

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

# Feature Selection
st.header("Feature Selection & Splitting")
st.write("The model uses the cleaned employee details, dummy-encoding columns for 'Department' and 'salary'.")
col_feat1, col_feat2 = st.columns(2)
with col_feat1:
    st.markdown("**Selected Features:**")
    st.write(list(df_clean.drop('left', axis=1).columns))
with col_feat2:
    st.markdown("**Train-Test Split dimensions (80% / 20%):**")
    st.write(f"Training Rows: `{X_train.shape[0]}` | Testing Rows: `{X_test.shape[0]}`")

st.markdown("---")

# Model Performance
st.header("Model Performance")

# Show KPIs in columns
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{len(df_raw)}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Cleaned Records</div>
        <div class="metric-value">{len(df_clean)}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Logistic Regression Accuracy</div>
        <div class="metric-value">{accuracy * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
                xticklabels=['Stayed (0)', 'Left (1)'],
                yticklabels=['Stayed (0)', 'Left (1)'])
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    fig3.tight_layout()
    st.pyplot(fig3)

with perf_col2:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Left (1)'])
    st.code(report, language='text')

st.markdown("---")

# Predict Section
st.header("Predict Employee Retention")
st.write("Use the form below to enter current employee details and evaluate their probability of staying or leaving the organization.")

# Prediction inputs in 2 columns
pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    satisfaction = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    last_evaluation = st.slider("Last Evaluation Score", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    number_project = st.slider("Number of Projects", min_value=2, max_value=7, value=4, step=1)
    average_hours = st.slider("Average Monthly Hours", min_value=90, max_value=310, value=200, step=1)
    time_spend = st.slider("Time Spent at Company (Years)", min_value=2, max_value=10, value=3, step=1)

with pred_col2:
    work_accident = st.selectbox("Work Accident?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    promotion = st.selectbox("Promotion in Last 5 Years?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    
    # Department lists
    departments = ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management']
    department = st.selectbox("Department", options=departments)
    
    # Salary level lists
    salary = st.selectbox("Salary Level", options=['low', 'medium', 'high'])

# Predict button
if st.button("Predict Employee Retention", type="primary"):
    # Construct input vector matching the dummy encoding columns
    input_data = pd.DataFrame(0, index=[0], columns=X.columns)
    
    # Fill in numerical features
    input_data['satisfaction_level'] = satisfaction
    input_data['last_evaluation'] = last_evaluation
    input_data['number_project'] = number_project
    input_data['average_montly_hours'] = average_hours
    input_data['time_spend_company'] = time_spend
    input_data['Work_accident'] = work_accident[1]
    input_data['promotion_last_5years'] = promotion[1]
    
    # Department dummy
    dep_col = f'Department_{department}'
    if dep_col in input_data.columns:
        input_data[dep_col] = 1
        
    # Salary dummies
    if salary == 'low':
        input_data['salary_low'] = 1
    elif salary == 'medium':
        input_data['salary_medium'] = 1
    # 'high' is the baseline (drop_first=True), so both low and medium dummy indicators remain 0
        
    # Make prediction
    pred_prob = model.predict_proba(input_data)[0]
    pred_class = model.predict(input_data)[0]
    
    stay_prob = pred_prob[0] * 100
    leave_prob = pred_prob[1] * 100
    
    st.markdown("### Prediction Results")
    
    # Displays nice cards
    if pred_class == 0:
        st.success("✅ **Employee is likely to STAY in the company**")
    else:
        st.error("⚠️ **Employee is likely to LEAVE the company**")

    # Probability metrics in columns
    m1, m2 = st.columns(2)
    m1.metric("🟢 Probability of Staying", f"{stay_prob:.2f}%")
    m2.metric("🔴 Probability of Leaving", f"{leave_prob:.2f}%")

    # Progress bars showing confidence
    st.markdown("**Retention Confidence Visualizer**")
    st.progress(stay_prob / 100.0, text=f"Staying: {stay_prob:.2f}%")
    st.progress(leave_prob / 100.0, text=f"Leaving: {leave_prob:.2f}%")
