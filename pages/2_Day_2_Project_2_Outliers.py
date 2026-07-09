import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Page Config
st.set_page_config(
    page_title="NYC Airbnb Outlier Detector",
    page_icon="🧹",
    layout="wide"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🧹 Project 2: Percentile-based Outlier Detection</h2>
    <p>Explore outlier detection techniques using New York City Airbnb 2019 data. Select lower and upper percentile cutoffs to see how listings are filtered live.</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) == "pages" else current_dir
    file_path = os.path.join(root_dir, "data", "AB_NYC_2019.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path {file_path} not found.")
    df = pd.read_csv(file_path)
    return df[['id', 'name', 'host_name', 'neighbourhood_group', 'room_type', 'price', 'minimum_nights']]

try:
    df_raw = load_data()
    
    # Sidebar Sliders
    st.sidebar.header("Percentile Cutoffs")
    lower_percentile = st.sidebar.slider("Lower Quantile Cutoff (%)", 0.0, 5.0, 1.0, 0.1)
    upper_percentile = st.sidebar.slider("Upper Quantile Cutoff (%)", 95.0, 100.0, 99.9, 0.1)
    
    # Compute thresholds
    lower_val = df_raw['price'].quantile(lower_percentile / 100.0)
    upper_val = df_raw['price'].quantile(upper_percentile / 100.0)
    
    # Filter dataset
    df_cleaned = df_raw[(df_raw['price'] >= lower_val) & (df_raw['price'] <= upper_val)]
    df_outliers_lower = df_raw[df_raw['price'] < lower_val]
    df_outliers_upper = df_raw[df_raw['price'] > upper_val]
    df_outliers = pd.concat([df_outliers_lower, df_outliers_upper])
    
    # Display thresholds
    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Bound Price Threshold", f"${lower_val:.2f} ({lower_percentile}%)")
    col2.metric("Upper Bound Price Threshold", f"${upper_val:.2f} ({upper_percentile}%)")
    col3.metric("Outliers Detected / Removed", f"{len(df_outliers):,} / {len(df_raw):,}")
    
    st.write("---")
    
    # Chart comparison
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("### 🛑 Original Price Distribution (All Data)")
        fig_raw = px.box(
            df_raw, 
            y="price", 
            points="outliers",
            color_discrete_sequence=['#ef4444'],
            labels={"price": "Price ($)"}
        )
        fig_raw.update_layout(yaxis_title="Price ($)")
        st.plotly_chart(fig_raw, use_container_width=True)
        
    with c2:
        st.write("### ✅ Cleaned Price Distribution (Outliers Removed)")
        fig_clean = px.box(
            df_cleaned, 
            y="price", 
            points="outliers",
            color_discrete_sequence=['#10b981'],
            labels={"price": "Price ($)"}
        )
        fig_clean.update_layout(yaxis_title="Price ($)")
        st.plotly_chart(fig_clean, use_container_width=True)
        
    st.write("---")
    
    # Data View tabs
    tab1, tab2, tab3 = st.tabs(["Cleaned Dataset Sample", "Detected Outliers Sample", "Descriptive Stats Comparison"])
    
    with tab1:
        st.write("#### Cleaned Dataset Preview")
        st.dataframe(df_cleaned.head(100), use_container_width=True)
        
    with tab2:
        st.write("#### Removed Outliers Preview")
        if len(df_outliers) > 0:
            st.dataframe(df_outliers.head(100), use_container_width=True)
        else:
            st.info("No outliers detected with current bounds.")
            
    with tab3:
        st.write("#### Comparison Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Min Price", "Max Price", "Mean Price", "Median Price", "Standard Dev", "Count"],
            "Original Data": [
                f"${df_raw['price'].min():.2f}",
                f"${df_raw['price'].max():.2f}",
                f"${df_raw['price'].mean():.2f}",
                f"${df_raw['price'].median():.2f}",
                f"${df_raw['price'].std():.2f}",
                f"{len(df_raw):,}"
            ],
            "Cleaned Data": [
                f"${df_cleaned['price'].min():.2f}",
                f"${df_cleaned['price'].max():.2f}",
                f"${df_cleaned['price'].mean():.2f}",
                f"${df_cleaned['price'].median():.2f}",
                f"${df_cleaned['price'].std():.2f}",
                f"{len(df_cleaned):,}"
            ]
        })
        st.table(stats_df)

except Exception as e:
    st.error(f"Error loading NYC Airbnb dataset: {e}")
