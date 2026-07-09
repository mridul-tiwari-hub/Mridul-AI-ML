import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="Google Play Store Analytics",
    page_icon="📊",
    layout="wide"
)

# Stylized header
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
    <h2>📊 Project 1: Google Play Store Analytics</h2>
    <p>An interactive Exploratory Data Analysis (EDA) dashboard displaying installation patterns, ratings distributions, and characteristics of Google Play Store applications.</p>
</div>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    file_path = os.path.join("data", "googleplaystore_v2.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path {file_path} not found.")
    
    df = pd.read_csv(file_path)
    
    # Drop rows with null ratings
    df = df.dropna(subset=['Rating'])
    
    # Clean Installs (e.g. "10,000+" -> 10000)
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # Clean Price (e.g. "$4.99" -> 4.99)
    df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Clean Reviews
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Clean Size (convert e.g. "14M", "512k" -> float Megabytes)
    def clean_size(val):
        if not isinstance(val, str):
            return np.nan
        val = val.lower().strip()
        if 'm' in val:
            return float(val.replace('m', ''))
        elif 'k' in val:
            return float(val.replace('k', '')) / 1024.0
        return np.nan

    df['Size'] = df['Size'].apply(clean_size)
    
    # Drop records that ended up with crucial null values during parsing
    df = df.dropna(subset=['Size', 'Installs', 'Price', 'Reviews'])
    return df

try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filter Settings")
    
    categories = ['All'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("App Category", categories)
    
    type_options = ['All'] + sorted(df['Type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Price Type (Free / Paid)", type_options)
    
    rating_slider = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.1)
    
    # Filter logic
    filtered_df = df[df['Rating'] >= rating_slider]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]
        
    # KPI displays
    st.subheader("💡 Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Apps Count", f"{len(filtered_df):,}")
    kpi2.metric("Average Rating", f"{filtered_df['Rating'].mean():.2f} ⭐")
    kpi3.metric("Total Reviews Recieved", f"{filtered_df['Reviews'].sum():,.0f}")
    kpi4.metric("Avg Price", f"${filtered_df['Price'].mean():.2f}")
    
    st.write("---")
    
    # Visualization layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📈 App Rating Distribution")
        fig_rating = px.histogram(
            filtered_df, 
            x="Rating", 
            nbins=20, 
            color_discrete_sequence=['#6366f1'],
            marginal="box",
            labels={"Rating": "User Rating"}
        )
        fig_rating.update_layout(bargap=0.05)
        st.plotly_chart(fig_rating, use_container_width=True)
        
    with col2:
        st.write("### 📏 App Size vs Rating Correlation")
        fig_scatter = px.scatter(
            filtered_df, 
            x="Size", 
            y="Rating", 
            color="Category",
            hover_name="App",
            size="Reviews",
            opacity=0.6,
            labels={"Size": "Size (MB)", "Rating": "Rating"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("### 🏆 Top 10 Categories by Average Rating")
        cat_ratings = filtered_df.groupby('Category')['Rating'].mean().reset_index().sort_values(by='Rating', ascending=False).head(10)
        fig_cat = px.bar(
            cat_ratings, 
            x="Rating", 
            y="Category", 
            orientation='h', 
            color="Rating",
            color_continuous_scale="plasma",
            labels={"Rating": "Avg Rating", "Category": "Category"}
        )
        fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col4:
        st.write("### 👥 Content Rating Breakdown")
        content_counts = filtered_df['Content Rating'].value_counts().reset_index()
        content_counts.columns = ['Content Rating', 'Count']
        fig_pie = px.pie(
            content_counts, 
            values="Count", 
            names="Content Rating", 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

except Exception as e:
    st.error(f"Error loading or preparing play store data: {e}")
