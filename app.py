import streamlit as st

# Configure the Streamlit Page
st.set_page_config(
    page_title="Mridul AI-ML Portfolio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main title layout */
    .title-container {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }
    
    .title-text {
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle-text {
        font-size: 1.3rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Project card styling */
    .project-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(255, 255, 255, 0.12);
    }
    
    .day-badge {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    
    .project-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #f3f4f6;
    }
    
    .project-desc {
        font-size: 0.9rem;
        color: #9ca3af;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown("""
<div class="title-container">
    <div class="title-text">🚀 Mridul AI-ML Project Portfolio</div>
    <div class="subtitle-text">An interactive showcase of data science, machine learning models, and deep learning apps built over 6 days.</div>
</div>
""", unsafe_allow_html=True)

st.write("### 🧭 Project Directory")
st.info("Use the sidebar on the left to navigate between different projects. Each page represents a distinct topic and interactive application.")

# Define grid for cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 1</span>
        <div class="project-title">📊 Project 1: Google Play Store Analytics</div>
        <p class="project-desc">
            An interactive Exploratory Data Analysis (EDA) dashboard cleaning and analyzing Google Playstore applications. 
            Filter through content categories, user reviews, and app sizes to discover ratings distributions and correlations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 2</span>
        <div class="project-title">📈 Project 3: Canada Per Capita Income Prediction</div>
        <p class="project-desc">
            A simple Linear Regression model mapping historical per capita income in Canada. 
            Input a future year to predict and visualize the linear income trajectory and fitted trend line.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 4</span>
        <div class="project-title">🐶 Project 5: Cat vs Dog Classifier</div>
        <p class="project-desc">
            A machine learning model classifying uploaded images of cats and dogs. 
            Preprocesses your images on the fly to yield prediction probabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 5</span>
        <div class="project-title">👤 Project 7: Gender Recognition CNN</div>
        <p class="project-desc">
            A Convolutional Neural Network (CNN) deep learning application analyzing facial inputs 
            to predict male or female categories with specific confidence intervals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 7</span>
        <div class="project-title">🤖 Project 9: Custom AI Assistant</div>
        <p class="project-desc">
            A ChatGPT-like custom assistant powered by OpenAI GPT models. 
            Configure personas, adjust parameters, and paste your API key to test.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 6</span>
        <div class="project-title">🩻 Project 11: COVID-19 Chest X-Ray Detector</div>
        <p class="project-desc">
            A deep learning CNN model classifying chest X-Rays as COVID-19 or normal. 
            Preprocesses medical scans and displays visual predictions with confidence indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 7</span>
        <div class="project-title">💧 Project 13: Samsung Washing Machine Assistant (RAG)</div>
        <p class="project-desc">
            A technical support chatbot using a LangChain RAG pipeline to answer user questions 
            directly based on details from the official Samsung Washing Machine manual.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 2</span>
        <div class="project-title">🧹 Project 2: Percentile Outlier Detection</div>
        <p class="project-desc">
            A hands-on outlier detection visualizer using the NYC Airbnb 2019 dataset. 
            Tune percentile sliders dynamically to detect, highlight, and filter out price outliers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 3</span>
        <div class="project-title">💼 Project 4: HR Employee Retention Analyzer</div>
        <p class="project-desc">
            A Logistic Regression model analyzing key drivers of employee turnover (satisfaction, average hours, salary). 
            Input employee statistics to predict retention probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 4</span>
        <div class="project-title">🔮 Project 6: K-Means Clustering Visualizer</div>
        <p class="project-desc">
            An interactive visualizer running the K-Means clustering algorithm on Iris petal features. 
            Adjust cluster size (K) in real-time, view cluster partitions, and check the Elbow Plot.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 6</span>
        <div class="project-title">🌸 Project 8: Iris Flower Classifier (KNN)</div>
        <p class="project-desc">
            A K-Nearest Neighbors (KNN) model classifying Iris flower species from sepal/petal dimensions. 
            Includes slider controls and live probability gauges.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 3</span>
        <div class="project-title">🏠 Project 10: Insurance Sales Predictor</div>
        <p class="project-desc">
            A Logistic Regression model predicting customer purchase likelihood based on age. 
            Plots the fitted sigmoid S-curve alongside actual data points.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 3</span>
        <div class="project-title">🛡️ Project 15: Life Insurance Purchase Prediction</div>
        <p class="project-desc">
            A logistic regression model predicting whether a person will buy life insurance based on their age.
            Displays dataset overview and live model parameter attributes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 6</span>
        <div class="project-title">👁️ Project 12: Male/Female Eye Classifier</div>
        <p class="project-desc">
            A custom-trained CNN classification model predicting eye gender (Male/Female) 
            from cropped close-up eye images.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    st.markdown("""
    <div class="project-card">
        <span class="day-badge">Day 7</span>
        <div class="project-title">🏥 Project 14: Star Health Insurance Assistant (RAG)</div>
        <p class="project-desc">
            A customer service RAG chatbot powered by OpenAI, trained to answer queries 
            concerning Star Health policy coverages, exclusions, and claims.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer info
st.markdown("---")
st.caption("Developed by Mridul | Streamlit Deployment Hub | Day 1 to Day 7 Projects")
