# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
# Load the regression model
classifier = pickle.load(open('loan_prediction.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    Gender_Male = int(request.form['Gender_Male'])
    Married_Yes = int(request.form['Married_Yes'])
    Education_Not_Graduate = int(request.form['Education_Not_Graduate'])
    Self_Employed_Yes = int(request.form['Self_Employed'])
    Property_Area_Urban = int(request.form['Property_Area_Urban'])
    Credit_History = int(request.form['Credit_History'])
    applicant_income = int(request.form['applicant_income'])
    coapplicant_income = int(request.form['coapplicant_income'])
    Dependents = int(request.form['Dependents'])
    Loan_Amount_Term_Years = int(request.form['Loan_Amount_Term_Years'])
    LoanAmount = int(request.form['LoanAmount'])

    total_income = applicant_income + coapplicant_income

    if Property_Area_Urban == 1:
        (Property_Area_Urban, Property_Area_Semiurban) = (1, 0)
    else:
        (Property_Area_Urban, Property_Area_Semiurban) = (0, 1)

    data = [[Dependents, LoanAmount, Credit_History, total_income, Loan_Amount_Term_Years, Gender_Male,
             Married_Yes, Education_Not_Graduate, Self_Employed_Yes,
             Property_Area_Semiurban, Property_Area_Urban]]
    if LoanAmount == 0:
        statement = 'Loan amount is zero'
    else:
        my_prediction = classifier.predict(data)
        if my_prediction == 0:
            statement = "Loan might not be approved"
        else:
            statement = "Loan might be approved"

    return render_template('index.html', prediction_text='{}'.format(statement))


if __name__ == "__main__":
    app.run(debug=True)
