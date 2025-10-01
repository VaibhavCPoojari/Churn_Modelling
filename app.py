import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

model=load_model('model.h5',compile=False)

with open('onehot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", ohe.categories_[0].tolist())
gender = st.selectbox("Gender", le.classes_.tolist())
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=['Germany', 'Spain', 'France'])
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]


if prediction_prob > 0.5:
    st.error("The customer is likely to leave the bank.")
else:
    st.success("The customer is likely to stay with the bank.")