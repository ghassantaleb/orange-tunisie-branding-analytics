import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and features
def load_awareness_assets():
    model = joblib.load('notebooks/awareness_model.pkl')
    features = joblib.load('notebooks/awareness_model_features.pkl')
    return model, features

model, model_features = load_awareness_assets()

# KPI labels
kpi_labels = ['Reach', 'Impressions']

st.title("📊 Awareness KPI Predictor")
st.write("Fill in campaign details to predict Reach and Impressions.")

# Date selection with calendar and year
date_range = st.date_input("Campaign Duration", value=(datetime(2024, 6, 1), datetime(2024, 6, 15)), help="Select campaign start and end date")
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

day = start_date.day
month = start_date.month
weekday = start_date.weekday()
year = start_date.year

amount_spent = st.number_input("Amount Spent (USD)", min_value=0.0, value=100.0, step=10.0)
use_ad_set_budget = st.checkbox("Use Ad Set Budget")

platforms = ['Platforme_Facebook', 'Platforme_Instagram']
placements = [
    'Placement_Facebook Reels',
    'Placement_Facebook Stories',
    'Placement_Facebook profile feed',
    'Placement_Feed',
    'Placement_Instagram Stories',
    'Placement_Native, banner & interstitial'
]
devices = ['Device_In-app', 'Device_Mobile Web']

selected_platform = st.selectbox("Platform", platforms)
selected_placement = st.selectbox("Placement", placements)
selected_device = st.selectbox("Device", devices)

# Build input vector
input_data = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
input_data['day'] = day
input_data['month'] = month
input_data['weekday'] = weekday
if 'Amount spent' in input_data.columns:
    input_data['Amount spent'] = amount_spent
if 'Ad set budget type_Using ad set budget' in input_data.columns:
    input_data['Ad set budget type_Using ad set budget'] = int(use_ad_set_budget)
if selected_platform in input_data.columns:
    input_data[selected_platform] = 1
if selected_placement in input_data.columns:
    input_data[selected_placement] = 1
if selected_device in input_data.columns:
    input_data[selected_device] = 1

# Predict
if st.button("Predict KPIs"):
    try:
        raw_prediction = model.predict(input_data)
        prediction = np.clip(raw_prediction, 0, None)
        scale = amount_spent / 1000
        prediction[0][0] *= scale  # Reach
        prediction[0][1] *= scale  # Impressions

        # Optional: Enforce Reach ≤ Impressions
        if prediction[0][0] > prediction[0][1]:
            prediction[0][0] = prediction[0][1] * 0.95

        st.success(f"Reach: {prediction[0][0]:,.2f}")
        st.success(f"Impressions: {prediction[0][1]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")