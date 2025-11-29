# 🏦 Customer Churn Prediction & Revenue Impact Simulator (Fintech)

An industry-oriented, end-to-end machine learning project designed to analyze customer behavior, predict churn, and simulate revenue impact for targeted retention campaigns.

---

# 📌 1. Project Summary
This project predicts **which fintech customers are most likely to churn** and estimates the **financial impact** of retention campaigns using a business-focused ROI simulator built with Streamlit.

---

# 🚨 2. Problem Statement & Business Impact

Customer churn is one of the biggest challenges in fintech, where the cost of acquiring a customer is high.  
By predicting churn early and quantifying potential revenue loss, companies can:

- Target high-value customers more efficiently  
- Reduce overall churn  
- Improve customer lifetime value (LTV)  
- Make data-backed retention decisions  
- Measure ROI before executing a marketing campaign  

This project creates a system that does exactly that.

---

## 🛠️ Tech Stack & Tools

**Programming & ML**  
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)

**Visualization**  
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C9ABF?logo=python&logoColor=white)

**Frameworks & Apps**  
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)

**Tools**  
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)
![VSCode](https://img.shields.io/badge/VS%20Code-0078D4?logo=visualstudiocode&logoColor=white)
![Excel](https://img.shields.io/badge/Excel-217346?logo=microsoft-excel&logoColor=white)


![Status](https://img.shields.io/badge/Project-Fintech%20Churn%20Analytics-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

# 📊 3. Key EDA Insights (From `EDA.ipynb`)

The EDA revealed strong behavioral patterns:

### 🔹 **1. Inactive customers are more likely to churn**
- Customers with higher **recency_days** show significantly higher churn probability.

### 🔹 **2. Low transaction activity signals disengagement**
- Users with **low monthly_txn_count**  
- And **low monthly_revenue**  
are more likely to churn.

### 🔹 **3. Premium customers churn less**
- Premium users display higher retention and stronger engagement signals.

### 🔹 **4. Complaints increase churn likelihood**
- Even a single complaint in the last 6 months shows a visible spike in churn rate.

### 🔹 **5. Active app users churn less**
- Higher **avg_session_minutes** correlates with low churn.

### 🔹 **6. Correlation Heatmap**
Shows:
- **recency_days** has strongest positive correlation with churn  
- Usage-based features (revenue, sessions, txn_count) negatively correlate with churn  

These insights directly guide model feature selection and business strategy.

---

# 🖼️ 4. Demo Screenshots

### 📈 EDA Snapshot — *Recency vs Churn*
![EDA Recency](screenshots/Recency_Distribution_By_Churn.png)

### 📉 Model ROC Curve
![ROC Curve](screenshots/roc_curve.png)

### 🧮 Streamlit Dashboard — *Top-K Risk + ROI*
![Dashboard](screenshots/streamlit_dashboard.png)

### 📄 Sample Excel Export (Targeted Customers)
![Excel Preview](screenshots/targeted_customers.png)

---

# 🧠 5. Tech Stack

- **Python 3.10+**
- **Machine Learning:** Pandas, NumPy, Scikit-Learn  
- **Visualizations:** Matplotlib, Seaborn  
- **Deployment/UI:** Streamlit, Flask  
- **Utilities:** Joblib, Openpyxl  
- **Testing:** PyTest  

---

# ⚙️ 6. Quick Start

### Clone repository
```sh
git clone https://github.com/girishshenoy16/Churn-Fintech-Simulator.git
cd Churn-Fintech-Simulator
```

### Create virtual environment
```sh
python -m venv .venv
.\.venv\Scripts\activate
```

### Install dependencies
```sh
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# ▶️ 7. How to Run the Project

### 1️⃣ Generate synthetic data
```sh
python scripts/generate_synthetic.py
```

### 2️⃣ Preprocess data
```sh
python src/data_preprocessing.py --input data/raw/sample_raw.csv --output data/processed/train_features.csv
```

### 3️⃣ Train churn model
```sh
python src/train_model.py --input data/processed/train_features.csv --output models/churn_model.pkl
```

### 4️⃣ Evaluating the Model 
```sh
python src/evaluate_model.py
```

### 5️⃣ Testing the Model 
```sh
python -m pytest
```

### 6️⃣ Launch Streamlit App
```sh
streamlit run app/streamlit_app.py
```

---

# 🗂️ 8. Project Structure

```
Churn-Fintech-Simulator/
│── app/
│   └── streamlit_app.py
│
│── data/
│   ├── processed/
│   │   └── train_features.csv
│   │
│   └── raw/
│       └── sample_raw.csv
│
│── models/
│   └── best_model.pkl
│
│── notebooks/
│   └── EDA.ipynb
│
│── scripts/
│   └── generate_synthetic.py
│ 
│── src/
│   ├── api.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
│ 
│── tests/
│   └── test_predict.py
│
│── README.md
└── requirements.txt
```


---


## 📊 Results
- Model: Logistic Regression
- Example AUC: ~0.60–0.85
- Precision@TopK: 3–4× better than random  
- ROI Positive in most simulations

## ✨ Future Scope
- XGBoost model
- Deployment to AWS/GCP
- Real-time scoring via Kafka + Redis
- SHAP explainability
