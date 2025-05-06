# Streamlit App for Traffic Campaign Predictions (CTR + Clicks)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- Load Models and Features ---
ctr_model = joblib.load("notebooks/ctr_model.pkl")
clicks_model = joblib.load("notebooks/clicks_model.pkl")
features = joblib.load("notebooks/traffic_model_features.pkl")

st.title("📊 Predict CTR and Clicks for Traffic Campaign")

# --- Campaign Design Inputs ---
st.subheader("Campaign Configuration")
date_range = st.date_input("Campaign Duration", [datetime.date.today(), datetime.date.today() + datetime.timedelta(days=7)])
start_date, end_date = date_range
n_days = (end_date - start_date).days + 1

# Average date features for prediction
day = start_date.day
month = start_date.month
weekday = start_date.weekday()

amount_spent = st.number_input("Amount Spent (USD)", min_value=1.0, step=10.0)
ad_set_budget_using = st.checkbox("Using Ad Set Budget")

platform = st.selectbox("Platform", ["Facebook", "Instagram"])
placements = st.multiselect("Placements", [
    "Facebook Reels", "Facebook Stories", "Facebook profile feed", "Feed",
    "Instagram Stories", "Native/Banner/Interstitial"
])
device = st.selectbox("Device", ["Mobile", "Desktop"])

# --- Construct Input Features ---
input_dict = {
    "Amount spent": amount_spent,
    "day": day,
    "month": month,
    "weekday": weekday,
    "Ad set budget type_Using ad set budget": int(ad_set_budget_using),
    f"Platforme_{platform}": 1,
    f"Device_{device}": 1,
}

for place in placements:
    input_dict[f"Placement_{place}"] = 1

# Fill missing expected features with 0
data = {feature: input_dict.get(feature, 0) for feature in features}
X_input = pd.DataFrame([data])

# --- Predict ---
if st.button("Predict"):
    try:
        log_ctr_pred = ctr_model.predict(X_input)[0]
        ctr_pred = np.expm1(log_ctr_pred)

        clicks_scaled_pred = clicks_model.predict(X_input)[0]
        clicks_pred = clicks_scaled_pred * (amount_spent / 1000) * n_days
        ctr_pred = ctr_pred  # Already day-level, shown as-is

        st.success(f"📈 Predicted CTR: {ctr_pred:.2f}%")
        st.success(f"🖱️ Predicted Clicks over {n_days} days: {clicks_pred:.0f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
