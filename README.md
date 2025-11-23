
# Telco Customer Churn Prediction

## Overview

This project predicts whether a telecom customer is likely to **churn (leave the company)** or **continue the service** based on various customer attributes such as monthly charges, tenure, contract type, and internet service. The system includes:

* **Exploratory Data Analysis (EDA)** to understand churn patterns
* **Machine Learning Model** (Random Forest Classifier)
* **Flask Web Application** for real-time predictions


## How It Works
### Data Flow

1. The original Kaggle dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is cleaned and analyzed in the EDA notebook.
2. Processed data (`tel_churn.csv`) is used to train a RandomForest model.
3. The model is serialized and saved as `model.sav`.
4. `first_telc.csv` (raw reference dataset) is used by the Flask app to ensure consistent feature columns during one-hot encoding.

### Model Details

* **Algorithm:** RandomForestClassifier
* **Features:** 50 encoded columns (categorical + numeric)
* **Metrics used:** Accuracy, Precision, Recall, F1-score, ROC-AUC
* **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn

## Tech stack used

* **Python 3.11**
* **Flask** (Web Framework)
* **scikit-learn** (Modeling)
* **pandas & numpy** (Data handling)
* **matplotlib & seaborn** (Visualization)

## Scope for improvement

* Add input validation and error handling for user fields.
* Deploy via **Docker** or **Render** instead of Flask dev server.
* Replace HTML UI with **Streamlit** or **Gradio** frontend.
