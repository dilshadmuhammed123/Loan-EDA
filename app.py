from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open("LGBMRegressor_model.pkl","rb"))
x=pickle.load(open("Scaler.pkl","rb"))

status_map={"unemployed":0,"self-Employed":1,"Employed":2}
edu_map={"High School":0,"Associate":1,"Bachelor":2,"Master":3,"Docterate":4}
features_to_log1p = ['LoanAmount','MonthlyIncome','NetWorth']
columns_to_standardize = [
    'Age', 'CreditScore', 'LoanAmount', 'LoanDuration',
    'CreditCardUtilizationRate', 'LengthOfCreditHistory',
    'MonthlyIncome', 'NetWorth', 'InterestRate'
]



@app.route('/')
def home():
    return render_template('index.html',prediction_text='')

@app.route('/predict', methods=['post'])
def index():
    input_data ={
        "Age":int(request.form['Age']),
        "CreditScore":int(request.form["CreditScore"]),
        "EmploymentStatus":request.form["EmploymentStatus"],
        "EducationLevel":request.form["EducationLevel"],
        "LoanAmount":float(request.form["LoanAmount"]),
        "LoanDuration":int(request.form["LoanDuration"]),
        "CreditCardUtilizationRate":float(request.form["CreditCardUtilizationRate"]),
        "BankruptcyHistory":int(request.form["BankruptcyHistory"]),
        "PreviousLoanDefaults":int(request.form["PreviousLoanDefaults"]),
        "LengthOfCreditHistory":int(request.form["LengthOfCreditHistory"]),
        "MonthlyIncome":int(request.form["MonthlyIncome"]),
        "NetWorth":float(request.form["NetWorth"]),
        "InterestRate":float(request.form["InterestRate"])
        
        
    }
    input_df=pd.DataFrame([input_data])


#Encioding
    input_df["EmploymentStatus"] =input_df["EmploymentStatus"].map(status_map)
    input_df["EducationLevel"]=input_df["EducationLevel"].map(edu_map)
#log1p transform
    input_df[features_to_log1p]=input_df[features_to_log1p].apply(np.log1p)
#standardize
    input_df[columns_to_standardize]=x.transform(input_df[columns_to_standardize])


#predict
    risk_score=model.predict(input_df)[0]
    risk_score=round(risk_score,2)
    return render_template('index.html',prediction_text=risk_score)

if __name__=='__main__':
    app.run(debug=True)









