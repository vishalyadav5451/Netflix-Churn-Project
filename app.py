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
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 100, 30)
    watch_hours = st.number_input("Total Watch Hours", 0.0, 1000.0, 100.0)
    monthly_fee = st.number_input("Monthly Fee (₹)", 0.0, 2000.0, 499.0)
    last_login_days = st.number_input("Days Since Last Login", 0, 365, 10)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type", [0, 1, 2])
    device = st.selectbox("Device", [0, 1, 2, 3])
    payment_method = st.selectbox("Payment Method", [0, 1, 2])

st.divider()

col3, col4 = st.columns(2)

with col3:
    region = st.selectbox("Region", [0, 1, 2, 3, 4, 5])
    number_of_profiles = st.slider("Number of Profiles", 1, 5, 2)

with col4:
    favorite_genre = st.selectbox("Favorite Genre", [0, 1, 2, 3])
    avg_watch_time_per_day = st.number_input("Avg Watch Time/Day (hrs)", 0.0, 24.0, 2.0)

st.divider()

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Churn"):

    # Encoding
    gender_encoded = 1 if gender == "Female" else 0

    # Derived Features (IMPORTANT)
    engagement_score = avg_watch_time_per_day / 24
    inactive_flag = 1 if last_login_days > 30 else 0
    high_price_flag = 1 if monthly_fee > 500 else 0

    # Feature Order (same as training)
    input_data = [[
        age,
        gender_encoded,
        subscription_type,
        watch_hours,
        last_login_days,
        region,
        device,
        monthly_fee,
        payment_method,
        number_of_profiles,
        favorite_genre,
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