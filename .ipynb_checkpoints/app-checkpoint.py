import streamlit as st
import joblib
import pandas as pd

model = joblib.load("customer_churn_model.pkl")

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")

if st.button("Predict"):
    data = pd.DataFrame([[tenure, monthly]],
                        columns=["tenure","MonthlyCharges"])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.write("Customer will churn")
    else:
        st.write("Customer will stay")
