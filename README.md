# 🏢 Employee Retention Prediction App

A machine learning web application built with **Streamlit** that predicts whether an employee will stay or leave a company, based on HR analytics data.

---

## 🚀 Live Demo

Run locally:
```bash
streamlit run app.py
```
Visit: **http://localhost:8501**

---

## 📊 Features

- **Dataset Preview** — Interactive table of 14,999 HR records
- **Data Cleaning** — Automatically removes 3,008 duplicate rows
- **Exploratory Data Analysis (EDA)** — Salary & Department retention charts
- **Model Performance** — Accuracy, Confusion Matrix heatmap, Classification Report
- **Live Prediction** — Enter employee details and get real-time retention prediction

---

## 🤖 Model

| Detail | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Dataset | HR Analytics CSV (`HR_comma_sep.csv`) |
| Training Records | 9,592 (80% split) |
| Test Records | 2,399 (20% split) |
| Test Accuracy | **83.33%** |

---

## 📁 Dataset Features

| Column | Description |
|--------|-------------|
| `satisfaction_level` | Employee satisfaction score (0–1) |
| `last_evaluation` | Last performance evaluation score (0–1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Avg. monthly working hours |
| `time_spend_company` | Years at the company |
| `Work_accident` | Had a work accident (0/1) |
| `promotion_last_5years` | Promoted in last 5 years (0/1) |
| `Department` | Employee department |
| `salary` | Salary level (low/medium/high) |
| `left` | **Target** — Did the employee leave? (0=Stayed, 1=Left) |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** — Web dashboard
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — Machine learning (Logistic Regression)
- **Matplotlib / Seaborn** — Visualization

---

## 👤 Author

**Mridul** — [GitHub](https://github.com/mridul-tiwari-hub)
