import streamlit as st
import joblib
import pandas as pd


loaded = joblib.load("customer_churn_model.pkl")
if isinstance(loaded, dict):
    model = loaded['model']
else:
    model = loaded

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict if they are likely to churn.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col2:
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                     "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)


if st.button("Predict Churn"):
   
    mapping_yes_no = {"No": 0, "Yes": 1}
    mapping_gender = {"Female": 0, "Male": 1}
    mapping_multiple = {"No phone service": 0, "No": 1, "Yes": 2}
    mapping_internet = {"DSL": 0, "Fiber optic": 1, "No": 2}
    mapping_contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    mapping_payment = {"Electronic check": 0, "Mailed check": 1,
                       "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
    
    input_data = {
        "gender": mapping_gender[gender],
        "SeniorCitizen": mapping_yes_no[senior],
        "Partner": mapping_yes_no[partner],
        "Dependents": mapping_yes_no[dependents],
        "tenure": tenure,
        "PhoneService": mapping_yes_no[phone_service],
        "MultipleLines": mapping_multiple[multiple_lines],
        "InternetService": mapping_internet[internet_service],
        "OnlineSecurity": mapping_multiple[online_security],
        "OnlineBackup": mapping_multiple[online_backup],
        "DeviceProtection": mapping_multiple[device_protection],
        "TechSupport": mapping_multiple[tech_support],
        "StreamingTV": mapping_multiple[streaming_tv],
        "StreamingMovies": mapping_multiple[streaming_movies],
        "Contract": mapping_contract[contract],
        "PaperlessBilling": mapping_yes_no[paperless],
        "PaymentMethod": mapping_payment[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    df_numeric = pd.DataFrame([input_data])
    try:
        df_numeric = df_numeric[model.feature_names_in_]
    except AttributeError:
        pass

    pred = model.predict(df_numeric)
    prob = model.predict_proba(df_numeric)

    st.subheader("Prediction Result:")
    if pred[0] == 1:
        st.error(f" Customer is likely to churn! (Probability: {prob[0][1]*100:.2f}%)")
    else:
        st.success(f" Customer is likely to stay. (Probability: {prob[0][0]*100:.2f}%)")
    
  
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({"Stay": [prob[0][0]], "Churn": [prob[0][1]]})
    st.bar_chart(prob_df)