import streamlit as st 
import joblib
import numpy as np


scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("model.pkl")

st.title('Churn Prediction App')

st.divider()

st.write('Please enter the values and hit predict button for getting a prediction')

st.divider()

age = st.number_input('Enter age' , min_value=10, max_value=100, value=30)

Watch_hours = st.number_input('Watch Hours' , min_value=0, max_value=500, value=50)

last_login_days = st.number_input('Last Activty' , min_value=0, max_value=365, value=10)

engagement_score = st.number_input('Engagement Score' , min_value=0, max_value=100, value=0)

gender = st.selectbox('Enter the Gender',["Male","Female"])

st.divider()

predictbutton = st.button("predict")

if predictbutton:

    gender_selected = 1 if gender == "Female" else 0

    X = [age , gender_selected , Watch_hours , last_login_days , engagement_score]

    X1 = np.array([X])

    X_array = scaler.transform(X1)

    prediction = rf_model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted: {predicted}")

else :
    st.write("please enter the values and use predict button")

