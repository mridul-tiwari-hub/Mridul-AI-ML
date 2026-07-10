# 🌟 Mridul AI-ML Streamlit Portfolio

Welcome to the **Mridul AI-ML Portfolio**! This repository hosts a unified, premium multi-page Streamlit application that showcases a series of data science, analytics, and deep learning projects spanning 7 days of intensive machine learning applications.

## 🚀 Interactive Project Directory

Navigate the sidebar in the Streamlit app to explore:

### 📅 Day 1: Data Analysis & Visualization
*   **Project 1: Google Play Store Analytics**
    *   Interactive Exploratory Data Analysis (EDA) on the Google Playstore apps dataset.
    *   Filters by categories, installations, and content rating with dynamic Plotly distributions and reviews/ratings correlation analysis.

### 📅 Day 2: Analytics & Linear Modeling
*   **Project 2: Outlier Detection (NYC Airbnb)**
    *   Dynamic percentile-based outlier identification. Select outlier thresholds via sliders to clean Airbnb listing prices live.
*   **Project 3: Canada Per Capita Income Predictor**
    *   Linear Regression model trained on Canada's historical income data. Input target years to visualize predictions on the regression trend.

### 📅 Day 3: Logistic Regression & Classification
*   **Project 4: Employee Retention Prediction**
    *   Logistic Regression model predicting if an employee will stay or leave based on satisfaction, salary, department, and work metrics.
*   **Project 10: Insurance Sales Predictor**
    *   Logistic Regression model predicting customer purchase likelihood based on age. Shows interactive Plotly sigmoid S-curve overlaying actual data.

### 📅 Day 4: Clustering & Machine Learning
*   **Project 5: Cat vs Dog Classifier**
    *   Flattened image classifier model identifying uploaded cat and dog images.
*   **Project 6: K-Means Clustering Visualizer**
    *   Real-time clustering on the Iris dataset. Adjust the number of clusters ($K$) dynamically to explore clustering partitions and the Elbow method.

### 📅 Day 5: Deep Learning & Computer Vision
*   **Project 7: Gender Recognition CNN**
    *   Convolutional Neural Network (CNN) classifying uploaded face images into Male or Female categories with confidence levels.

### 📅 Day 6: Instance-Based & Deep Learning
*   **Project 8: Iris Flower Classifier (KNN)**
    *   K-Nearest Neighbors model classifying Iris species from user-adjusted sepal/petal dimensions.
*   **Project 11: COVID-19 Chest X-Ray Detector**
    *   A deep learning CNN model trained on chest scans to classify COVID-19 vs Normal status.
*   **Project 12: Male/Female Eye Classifier**
    *   A CNN model identifying eye gender from cropped eye images.

### 📅 Day 7: Large Language Models & RAG Chatbots
*   **Project 9: Custom AI Assistant (OpenAI GPT-4o)**
    *   A ChatGPT-like custom assistant powered by OpenAI APIs. Set custom personalities, temperature parameters, and query the assistant securely.
*   **Project 13: Samsung Washing Machine Assistant (RAG)**
    *   An assistant chatbot that utilizes LangChain and Chroma vector store to answer queries from the official Samsung Washing Machine HTML manual.
*   **Project 14: Star Health Insurance Assistant (RAG)**
    *   A customer service assistant running RAG over Star Health insurance policy documents to query claims, covers, and exclusions.

---

## 🛠️ Local Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/mridul-tiwari-hub/Mridul-AI-ML.git
    cd Mridul-AI-ML
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Linux/macOS: source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

---

## ☁️ Streamlit Cloud Deployment
This structure is 100% compliant with Streamlit Community Cloud. To deploy:
1. Push to your GitHub repository.
2. Link the repository to your [Streamlit Share account](https://share.streamlit.io/).
3. Set the entry point script to `app.py`.
