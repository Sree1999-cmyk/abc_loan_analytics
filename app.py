from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the models and scalers
cls = joblib.load('logistic_regression_model.pkl')
std_cls = joblib.load('standard_scaler_cls.pkl')
lin_model = joblib.load('linear_regression_model.pkl')
std_lin = joblib.load('standard_scaler_lin.pkl')

# Define the columns that the models expect (based on X and X_lin from the notebook)
# These should be consistent with the training data columns after encoding.
# Assuming X.columns and X_lin.columns were captured during training.
# For simplicity, let's hardcode them based on the notebook's X and X_lin.
# In a real app, you might save these alongside the models.
CLASSIFICATION_FEATURES = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file'
]

REGRESSION_FEATURES = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'person_home_ownership', 'loan_intent',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file'
]

@app.route('/predict_loan_status', methods=['POST'])
def predict_loan_status():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([data])

        # Ensure the input DataFrame has the correct columns and order
        df_input = df_input[CLASSIFICATION_FEATURES]

        # Scale the input data using the scaler trained for classification
        scaled_data = std_cls.transform(df_input)

        # Make prediction
        prediction = cls.predict(scaled_data)[0]
        status = 'Approved' if prediction == 1 else 'Rejected'

        return jsonify({'loan_status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_loan_amount', methods=['POST'])
def predict_loan_amount():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([data])

        # Ensure the input DataFrame has the correct columns and order
        df_input = df_input[REGRESSION_FEATURES]

        # Scale the input data using the scaler trained for regression
        scaled_data = std_lin.transform(df_input)

        # Make prediction
        prediction = lin_model.predict(scaled_data)[0]

        return jsonify({'predicted_loan_amount': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For development, run with debug=True. For production, use a production WSGI server.
    app.run(host='0.0.0.0', port=5000, debug=True)
