# to run 
# python app.py

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1=pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    # Collect all form inputs
    input_values = [request.form[f'query{i}'] for i in range(1, 20)]

    # Load model once per request (you can move this to top-level if you prefer)
    model = pickle.load(open("model.sav", "rb"))

    # Create a new DataFrame for the input
    new_df = pd.DataFrame([input_values], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Combine with reference dataset for consistent dummy columns
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12),
                                  right=False, labels=labels)
    df_2.drop(columns=['tenure'], inplace=True)

    # One-hot encode categorical columns
    new_df_dummies = pd.get_dummies(df_2[[
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]])

    df_2 = df_2.replace(' ', None)

    # Convert columns to numeric safely
    df_2['SeniorCitizen'] = pd.to_numeric(df_2['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    df_2['MonthlyCharges'] = pd.to_numeric(df_2['MonthlyCharges'], errors='coerce').fillna(0)
    df_2['TotalCharges']  = pd.to_numeric(df_2['TotalCharges'],  errors='coerce').fillna(0)


    # Align columns with model’s training features
    new_df_dummies = new_df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

    if single == 1:
        o1 = "This customer is likely to be churned!!"
    else:
        o1 = "This customer is likely to continue!!"
    o2 = f"Confidence: {probability[0]*100:.2f}%"

    return render_template(
        'home.html', output1=o1, output2=o2,
        **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)}
    )
    
app.run()

