# Streamlit App for CTR Predictions (Classification vs. Ratio)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- Load Models and Features ---
ctr_classifier = joblib.load("notebooks/ctr_classifier.pkl")
click_model = joblib.load("notebooks/clicks_model.pkl")
impr_model = joblib.load("notebooks/impressions_model.pkl")
features = joblib.load("notebooks/traffic_model_features.pkl")

st.title("📊 Compare CTR Prediction Approaches")

# --- Campaign Design Inputs ---
st.subheader("Campaign Configuration")
date_range = st.date_input("Campaign Duration", [datetime.date.today(), datetime.date.today() + datetime.timedelta(days=7)])
start_date, end_date = date_range
n_days = (end_date - start_date).days + 1

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
if st.button("Predict CTR"):
    try:
        # Ratio-based CTR
        clicks_scaled_pred = click_model.predict(X_input)[0]
        impr_scaled_pred = impr_model.predict(X_input)[0]
        ctr_ratio = (clicks_scaled_pred / impr_scaled_pred) * 100

        # Classification-based CTR
        ctr_class = ctr_classifier.predict(X_input)[0]
        class_label = {0: "Low (<0.5%)", 1: "Medium (0.5% - 1.5%)", 2: "High (>1.5%)"}[ctr_class]

        st.markdown("### 🧠 CTR Predictions:")
        st.success(f"🔢 CTR (Ratio Model): {ctr_ratio:.2f}%")
        st.info(f"🏷️ CTR (Classification Model): {class_label}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
