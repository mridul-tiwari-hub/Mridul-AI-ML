import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="K-Means Clustering Visualizer",
    page_icon="🔮",
    layout="wide"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🔮 Project 6: K-Means Clustering Visualizer</h2>
    <p>Perform unsupervised learning on the Iris flower dataset using Petal Length and Petal Width. Adjust cluster parameters to visualize centroids and explore the Elbow method.</p>
</div>
""", unsafe_allow_html=True)

try:
    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Keep only Petal features as in homework
    df_petal = df[['petal length (cm)', 'petal width (cm)']].copy()
    
    # Sidebar parameter selection
    st.sidebar.header("K-Means Hyperparameters")
    k = st.sidebar.slider("Number of Clusters (K)", 1, 10, 3, 1)
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_petal)
    df_scaled = pd.DataFrame(scaled_data, columns=['petal_length', 'petal_width'])
    
    # Run KMeans
    km = KMeans(n_clusters=k, random_state=42)
    df_petal['cluster'] = km.fit_predict(df_scaled)
    centroids = km.cluster_centers_
    
    # Map clusters back to scaled space or original space
    centroids_orig = scaler.inverse_transform(centroids)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### 🍀 Cluster Partitions (K = {k})")
        
        colors = px.colors.qualitative.Plotly
        fig = go.Figure()
        
        # Add clustered data points
        for cluster_id in range(k):
            cluster_data = df_petal[df_petal['cluster'] == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_data['petal length (cm)'],
                y=cluster_data['petal width (cm)'],
                mode='markers',
                name=f'Cluster {cluster_id + 1}',
                marker=dict(color=colors[cluster_id % len(colors)], size=10, opacity=0.7)
            ))
            
        # Add centroids
        fig.add_trace(go.Scatter(
            x=centroids_orig[:, 0],
            y=centroids_orig[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(color='black', size=15, symbol='x', line=dict(color='white', width=2))
        ))
        
        fig.update_layout(
            xaxis_title="Petal Length (cm)",
            yaxis_title="Petal Width (cm)",
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write("### 📐 The Elbow Method (Inertia Plot)")
        
        # Calculate SSE for k=1..10
        sse = []
        k_range = range(1, 11)
        for i in k_range:
            km_temp = KMeans(n_clusters=i, random_state=42)
            km_temp.fit(df_scaled)
            sse.append(km_temp.inertia_)
            
        # Plot Elbow curve
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range),
            y=sse,
            mode='lines+markers',
            line=dict(color='#be185d', width=2),
            marker=dict(size=8),
            name='SSE (Inertia)'
        ))
        
        # Highlight current K
        fig_elbow.add_trace(go.Scatter(
            x=[k],
            y=[sse[k-1]],
            mode='markers',
            name='Current K Selection',
            marker=dict(color='black', size=12, symbol='circle')
        ))
        
        fig_elbow.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Sum of Squared Errors (SSE)",
            legend=dict(x=0.6, y=0.99),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    st.write("---")
    st.write("### 📋 Clustered Dataset Snippet")
    st.dataframe(df_petal.head(100), use_container_width=True)
    
except Exception as e:
    st.error(f"Error executing K-Means visualization: {e}")
