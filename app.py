import streamlit as st

# MUST be first
st.set_page_config(page_title="Netflix Churn Predictor", page_icon="🎬")

import joblib
import numpy as np

# ===============================
# Load Pipeline (Scaler + Model)
# ===============================
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("pipeline.pkl")
    return pipeline

pipeline = load_pipeline()

# ===============================
# UI
# ===============================
st.title("🎬 Netflix Customer Churn Prediction")
st.markdown("Fill customer details to predict churn probability")

st.divider()

# ===============================
# INPUTS
# ===============================
subscription_map = {
    'Basic' : 0,
    'Standard' : 1 ,
    'Premium' : 2
}
device_map = {
    'Desktop': 0 , 'Laptop': 1, 'Mobile': 2, 'TV': 3, 'Tablet': 4
}
payment_map = {
     'Credit Card':0, 'Crypto':1, 'Debit Card':2, 'Gift Card':3, 'PayPal':4
 }
region_map = {
    'Africa':0, 'Asia':1, 'Europe':2, 'North America':3, 'Oceania':4,
       'South America':5
}
genre_map = {
    'Action':0, 'Comedy':1, 'Documentary':2, 'Drama':3, 'Horror':4, 'Romance':5,
       'Sci-Fi':6
}
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 100, 30)
    watch_hours = st.number_input("Total Watch Hours", 0.0, 1000.0, 100.0)
    monthly_fee = st.number_input("Monthly Fee (₹)", 0.0, 2000.0, 499.0)
    last_login_days = st.number_input("Days Since Last Login", 0, 365, 10)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type",list(subscription_map.keys()))
    device = st.selectbox("Device", list(device_map.keys()))
    payment_method = st.selectbox("Payment Method", list(payment_map.keys()))
    

st.divider()

col3, col4 = st.columns(2)

with col3:
    region = st.selectbox("Region", list(region_map.keys()))
    number_of_profiles = st.slider("Number of Profiles", 1, 5, 2)

with col4:
    favorite_genre = st.selectbox("Favorite Genre", list(genre_map.keys()))
    avg_watch_time_per_day = st.number_input("Avg Watch Time/Day (hrs)", 0.0, 24.0, 2.0)

st.divider()


# ===============================
# PREDICTION
# ===============================
if st.button("Predict Churn"):

    # Encoding
    gender_encoded = 1 if gender == "Female" else 0
    subscription_encoded = subscription_map[subscription_type]
    device_encoded = device_map[device]
    payment_encoded = payment_map[payment_method]
    region_encoded = region_map[region]
    genre_encoded = genre_map[favorite_genre]

    # Derived Features (IMPORTANT)
    engagement_score = avg_watch_time_per_day / 24
    inactive_flag = 1 if last_login_days > 30 else 0
    high_price_flag = 1 if monthly_fee > 500 else 0

    # Feature Order (same as training)
    input_data = [[
        age,
        gender_encoded,
        subscription_encoded,
        watch_hours,
        last_login_days,
        region_encoded,
        device_encoded,
        monthly_fee,
        payment_encoded,
        number_of_profiles,
        genre_encoded,
        engagement_score,
        inactive_flag,
        high_price_flag
    ]]

    X = np.array(input_data)

    # ✅ Pipeline handles scaling + prediction
    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0][1]

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is NOT likely to churn")

    st.write(f"Churn Probability: {probability:.2%}")

    # Risk Level
    if probability > 0.7:
        st.error("High Risk Customer")
    elif probability > 0.4:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")

    # Debug
    with st.expander("Debug Info"):
        st.write("Input Data:", input_data)
        st.write("Prediction Probability:", probability)

else:
    st.info("Enter customer details and click Predict Churn.")

# ===============================
# TEST MODE (FOR VALIDATION)
# ===============================
# st.divider()
# st.subheader("🧪 Test Mode")

# if st.button("Test Known Churn Case"):

#     # Direct scaled test (same as notebook)
#     X_scaled = np.array([[ 
#         1.5, 0, 1, -1.2, 60, 3, 2, 1.5, 1, 0.3, 2, 0.1, 1, 1
#     ]])

#     # Access model inside pipeline
#     model = pipeline.named_steps['model']

#     prediction = model.predict(X_scaled)[0]
#     prob = model.predict_proba(X_scaled)[0][1]

#     st.write("Prediction:", prediction)
#     st.write("Probability:", prob)