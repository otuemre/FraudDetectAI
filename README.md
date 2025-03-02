# FraudDetectAI - A Fraud Detection AI

## Project Overview
FraudDetectAI is a machine learning-based system designed to **detect fraudulent credit card transactions**.  
Using **imbalanced learning techniques**, feature engineering, and powerful models like **XGBoost**, this project aims to build a **highly accurate fraud detection system**.

---

## Objectives
- Understand the nature of fraudulent transactions.  
- Handle **highly imbalanced data** effectively.  
- Build a **robust classification model** for fraud detection.  
- Use **explainability techniques** (SHAP) to interpret model predictions.

---

## Steps We Will Cover

### **1️⃣ Exploratory Data Analysis (EDA)**
- Dataset overview and missing value analysis.  
- Fraud vs. Non-Fraud transaction distribution.  
- Transaction amount & time distribution analysis.  
- PCA feature importance analysis (V1-V28).  
- Identified **top 5 PCA features** for fraud detection.  

### **2️⃣ Data Preprocessing & Feature Engineering**
- Handling class imbalance (SMOTE, undersampling, etc.).  
- Feature scaling (standardization or normalization).  
- Feature selection to improve model efficiency.  

### **3️⃣ Model Training & Evaluation**
- Trying **XGBoost, Random Forest, and Anomaly Detection**.  
- Hyperparameter tuning using **Optuna**.  
- Evaluating **precision-recall, AUC-ROC** due to class imbalance.  

### **4️⃣ Model Explainability & Finalization**
- Using **SHAP** to understand feature importance.  
- Deploying the final fraud detection model.  

---

## Findings So Far

- **Fraud transactions are extremely rare (0.17%)**, making imbalance handling crucial.  
- Fraud transactions occur **more often at night (1 AM - 6 AM)**.  
- Certain **PCA features (V17, V14, V12, V10, V16) strongly correlate with fraud**.  
- Fraud transactions **tend to have different distributions** in key features.  

---

## Next Steps
1. Handle **class imbalance** using resampling techniques.  
2. **Scale and preprocess features** for better model performance.  
3. Train **machine learning models** and evaluate performance.  

---

## Project Structure
```
FraudDetectAI/
│── src/
│   ├── datasets/           # Raw and processed datasets
│   ├── models/             # Trained models
│   ├── notebooks/          # Jupyter notebooks for analysis
│   ├── images/             # Saved visualizations
│   ├── reports/            # Project reports & summaries
│── .gitignore              # Ignore the uncessary files
│── README.md               # Project documentation
│── requirements.txt        # Dependencies
```

---

## Acknowledgments
This project is based on the **Credit Card Fraud Detection dataset** from [mlg-ulb](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
