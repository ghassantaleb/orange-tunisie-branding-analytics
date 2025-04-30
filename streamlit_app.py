import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and expected feature columns
model = joblib.load("C:/Users/ghass/orange-tunisie-branding-analytics/models/rf_ctr_traffic_model.joblib")
expected_features = joblib.load("models/traffic_model_features.pkl")

# Streamlit app title
st.title("Predict CTR for Traffic Campaigns")
st.markdown("""
This tool uses a trained Random Forest model to predict **CTR (%)**  
based on your campaign inputs (platform, placement, device, budget...).
""")

# --- User inputs ---
st.header("Enter Campaign Settings")

budget = st.number_input("Ad Set Budget (USD)", min_value=1.0, value=50.0)
platform = st.selectbox("Platform", ["Facebook", "Instagram", "Audience Network", "Messenger"])
device = st.selectbox("Device", ["Mobile", "Desktop"])
placement = st.selectbox("Placement", ["Feed", "Stories", "In-Stream", "Search", "Other"])
budget_type = st.selectbox("Budget Type", ["daily", "lifetime"])
day = st.slider("Day of Month", 1, 31, value=15)
month = st.slider("Month", 1, 12, value=4)
weekday = st.slider("Weekday (0=Mon)", 0, 6, value=2)

# --- Feature Encoding ---
def prepare_features():
    # Your basic input dictionary
    input_data = {
        "log_spend": np.log1p(budget),
        "day": day,
        "month": month,
        "weekday": weekday,
        "Platforme_Instagram": int(platform == "Instagram"),
        "Platforme_Messenger": int(platform == "Messenger"),
        "Platforme_Audience Network": int(platform == "Audience Network"),
        "Placement_Stories": int(placement == "Stories"),
        "Placement_In-Stream": int(placement == "In-Stream"),
        "Placement_Search": int(placement == "Search"),
        "Placement_Other": int(placement == "Other"),
        "Device_Mobile": int(device == "Mobile"),
        "Ad set budget type_lifetime": int(budget_type == "lifetime")
    }

    # Create DataFrame and align to training columns
    input_df = pd.DataFrame([input_data])

    # Add missing columns as zeros
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match model
    input_df = input_df[expected_features]

    return input_df


# --- Predict ---
if st.button("Predict CTR"):
    X_input = prepare_features()
    log_ctr = model.predict(X_input)[0]
    ctr = np.expm1(log_ctr)  # convert from log scale
    st.success(f"Predicted CTR: **{ctr:.2f}%**")
