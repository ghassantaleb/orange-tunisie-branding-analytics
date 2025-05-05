# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer

# Load models and transformers
model = joblib.load("C:/Users/ghass/orange-tunisie-branding-analytics/notebooks/models/bayesian_ridge_traffic_model.joblib")
power = joblib.load("C:/Users/ghass/orange-tunisie-branding-analytics/notebooks/models/traffic_power_transformer.joblib")
feature_names = joblib.load("C:/Users/ghass/orange-tunisie-branding-analytics/notebooks/models/traffic_model_features.pkl")

st.set_page_config(page_title="Traffic Campaign Predictor", layout="centered")
st.title("📊 Traffic Campaign KPI Predictor")
st.markdown("Enter campaign design details to predict **CTR (all)** and **Link Clicks**.")

# User inputs
amount_spent = st.number_input("Amount Spent", min_value=0.0, step=0.1)
reach = st.number_input("Reach", min_value=0.0, step=1.0)
impressions = st.number_input("Impressions", min_value=0.0, step=1.0)
clicks_all = st.number_input("Clicks (all)", min_value=0.0, step=1.0)
ctr_link = st.number_input("CTR (link click-through rate)", min_value=0.0, step=0.01)
cpc_all = st.number_input("CPC (all) (USD)", min_value=0.0, step=0.01)
landing_views = st.number_input("Landing Page Views", min_value=0.0, step=1.0)
cplpv = st.number_input("Cost per Landing Page View (USD)", min_value=0.0, step=0.01)
link_clicks = st.number_input("Link Clicks", min_value=0.0, step=1.0)
unique_link_clicks = st.number_input("Unique Link Clicks", min_value=0.0, step=1.0)
unique_clicks_all = st.number_input("Unique Clicks (all)", min_value=0.0, step=1.0)
unique_ctr_all = st.number_input("Unique CTR (all)", min_value=0.0, step=0.01)
unique_ctr_link = st.number_input("Unique CTR (link click-through rate)", min_value=0.0, step=0.01)
cpc_link = st.number_input("CPC (cost per link click) (USD)", min_value=0.0, step=0.01)

# Categorical inputs
device = st.selectbox("Device", ["In-app", "None"])
placement_options = [
    "Facebook Reels", "Facebook Stories", "Facebook profile feed",
    "Feed", "Instagram Stories", "Native, banner & interstitial"
]
selected_placements = st.multiselect("Placements", placement_options)

# Build input DataFrame
input_data = {
    "Amount spent": amount_spent,
    "Reach": reach,
    "Impressions": impressions,
    "Clicks (all)": clicks_all,
    "CTR (link click-through rate)": ctr_link,
    "CPC (all) (USD)": cpc_all,
    "Landing page views": landing_views,
    "Cost per landing page view (USD)": cplpv,
    "Link clicks": link_clicks,
    "Unique link clicks": unique_link_clicks,
    "Unique clicks (all)": unique_clicks_all,
    "Unique CTR (all)": unique_ctr_all,
    "Unique CTR (link click-through rate)": unique_ctr_link,
    "CPC (cost per link click) (USD)": cpc_link
}

# Manually encode one-hot features
for placement in placement_options:
    input_data[f"Placement_{placement}"] = 1 if placement in selected_placements else 0

input_data["Device_In-app"] = 1 if device == "In-app" else 0

# Ensure all features exist and are ordered
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0  # fill missing one-hot features

X_input = pd.DataFrame([input_data])[feature_names]

# Predict
if st.button("Predict KPIs"):
    try:
        X_transformed = power.transform(X_input)
        prediction = model.predict(X_transformed)
        ctr_pred, clicks_pred = prediction[0]

        st.success("✅ Prediction complete!")
        st.metric("Predicted CTR (all)", f"{ctr_pred:.4f}")
        st.metric("Predicted Link Clicks", f"{clicks_pred:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")