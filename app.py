import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

st.set_page_config(page_title="ABC Ltd Loan Predictor", layout="centered")

st.title("ABC Ltd: Loan Approval & Amount Predictor")
st.write("Enter applicant details below to run Logistic Regression (Loan Approval) and Linear Regression (Loan Amount).")

@st.cache_resource
def build_models():
    data = pd.read_csv("loan_data.csv")
    data = data[(data['person_age'] <= 80) & (data['person_income'] <= 150000)].copy()

    text_cols = [
        'person_gender', 'person_education', 'person_home_ownership',
        'loan_intent', 'previous_loan_defaults_on_file'
    ]
    for col in text_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Model 1: Logistic Regression (Target: loan_status)
    X_cls = data.drop(['loan_status'], axis=1)
    y_cls = data['loan_status']
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    sc_cls = StandardScaler()
    X_tr_c_scaled = sc_cls.fit_transform(X_tr_c)
    X_te_c_scaled = sc_cls.transform(X_te_c)
    
    clf = LogisticRegression()
    clf.fit(X_tr_c_scaled, y_tr_c)
    cls_acc = clf.score(X_te_c_scaled, y_te_c)

    # Model 2: Linear Regression (Target: loan_amnt)
    X_reg = data.drop(['loan_amnt', 'loan_status'], axis=1)
    y_reg = data['loan_amnt']
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    sc_reg = StandardScaler()
    X_tr_r_scaled = sc_reg.fit_transform(X_tr_r)
    X_te_r_scaled = sc_reg.transform(X_te_r)
    
    reg = LinearRegression()
    reg.fit(X_tr_r_scaled, y_tr_r)
    reg_r2 = reg.score(X_te_r_scaled, y_te_r)

    return clf, sc_cls, X_cls.columns, cls_acc, reg, sc_reg, X_reg.columns, reg_r2

clf, sc_cls, cls_cols, cls_acc, reg, sc_reg, reg_cols, reg_r2 = build_models()

st.info(f"Logistic Regression Accuracy: {cls_acc*100:.2f}% | Linear Regression R² Score: {reg_r2*100:.2f}%")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 80, 28)
    gender = st.selectbox("Gender", ["female", "male"])
    edu = st.selectbox("Education", ["Associate", "Bachelor", "Doctorate", "High School", "Master"], index=1)
    home = st.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"], index=3)
    exp = st.number_input("Experience (Years)", 0, 50, 4)
    cred_len = st.number_input("Credit History (Years)", 2, 30, 4)

with col2:
    inc = st.number_input("Annual Income ($)", 8000, 150000, 60000, step=1000)
    req_loan = st.number_input("Requested Loan ($)", 500, 35000, 12000, step=500)
    intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"], index=4)
    rate = st.slider("Interest Rate (%)", 5.4, 20.0, 11.0, 0.1)
    score = st.slider("Credit Score", 390, 850, 650)
    default = st.selectbox("Prior Default on File?", ["No", "Yes"])

ratio = round(req_loan / inc, 2)
st.caption(f"Loan-to-Income Ratio: {ratio}")

g_map = {"female": 0, "male": 1}
e_map = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
h_map = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
i_map = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2, "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5}
d_map = {"No": 0, "Yes": 1}

if st.button("Predict Loan Outcome", use_container_width=True):
    row = {
        'person_age': [float(age)],
        'person_gender': [g_map[gender]],
        'person_education': [e_map[edu]],
        'person_income': [float(inc)],
        'person_emp_exp': [int(exp)],
        'person_home_ownership': [h_map[home]],
        'loan_amnt': [float(req_loan)],
        'loan_intent': [i_map[intent]],
        'loan_int_rate': [float(rate)],
        'loan_percent_income': [float(ratio)],
        'cb_person_cred_hist_length': [float(cred_len)],
        'credit_score': [int(score)],
        'previous_loan_defaults_on_file': [d_map[default]]
    }
    
    df_c = pd.DataFrame(row)[cls_cols]
    # In credit risk data: class 0 = Safe/Low Risk (Approve), class 1 = Default Risk (Reject)
    safe_prob = clf.predict_proba(sc_cls.transform(df_c))[0][0]

    df_r = pd.DataFrame(row).drop(['loan_amnt'], axis=1)[reg_cols]
    pred_amt = max(500.0, reg.predict(sc_reg.transform(df_r))[0])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Logistic Regression")
        if default == "Yes":
            st.error("REJECTED (Reason: Prior Loan Default on File)")
        elif safe_prob >= 0.50:
            st.success(f"APPROVED - Low Risk (Confidence: {safe_prob*100:.1f}%)")
        else:
            st.error(f"REJECTED - High Default Risk (Approval Prob: {safe_prob*100:.1f}%)")
    with c2:
        st.subheader("Linear Regression")
        st.metric("Sanctioned Loan Amount", f"${pred_amt:,.2f}")
