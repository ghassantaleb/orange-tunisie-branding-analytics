import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models, features, and transformers
@st.cache_resource
def load_assets():
    models = {
        'Awareness': joblib.load('notebooks/models/random_forest_awareness_model.joblib'),
        'Engagement': joblib.load('notebooks/models/random_forest_engagement_model.joblib'),
        'Traffic': joblib.load('notebooks/models/random_forest_traffic_model.joblib')
    }
    features = {
        'Awareness': joblib.load('models/awareness_model_features.pkl'),
        'Engagement': joblib.load('models/engagement_model_features.pkl'),
        'Traffic': joblib.load('models/traffic_model_features.pkl')
    }
    transformers = {
        'Awareness': joblib.load('notebooks/models/awareness_power_transformer.joblib'),
        'Engagement': joblib.load('notebooks/models/engagement_power_transformer.joblib'),
        'Traffic': joblib.load('notebooks/models/traffic_power_transformer.joblib')
    }
    return models, features, transformers

models, features, transformers = load_assets()

# KPI Labels
kpi_labels = {
    'Awareness': ['Reach', 'Impressions'],
    'Engagement': ['Post Engagements', 'Page Engagement'],
    'Traffic': ['CTR', 'Clicks']
}

# UI Title
st.title("📊 Campaign KPI Predictor")
st.write("Select a campaign type and fill in campaign setup details to predict key performance indicators (KPIs).")

# Campaign type selection
campaign_type = st.selectbox("Choose Campaign Objective", list(models.keys()))
selected_model = models[campaign_type]
model_features = features[campaign_type]
transformer = transformers[campaign_type]

# Input placeholders
st.header("Campaign Design Inputs")

# Time Features
day = st.slider("Day of Month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
weekday = st.selectbox("Weekday (0=Monday, 6=Sunday)", list(range(7)))

# Amount Spent
amount_spent = st.number_input("Amount Spent (USD)", min_value=0.0, value=100.0, step=10.0)

# Ad set budget type
use_ad_set_budget = st.checkbox("Use Ad Set Budget")

# Platform Options
platforms = ['Platforme_Facebook', 'Platforme_Instagram']
selected_platform = st.selectbox("Platform", platforms)

# Placement Options (unified subset)
placements = [
    'Placement_Facebook Reels',
    'Placement_Facebook Stories',
    'Placement_Facebook profile feed',
    'Placement_Feed',
    'Placement_Instagram Stories',
    'Placement_Native, banner & interstitial'
]
selected_placement = st.selectbox("Placement", placements)

# Device Options
devices = ['Device_In-app', 'Device_Mobile Web']
selected_device = st.selectbox("Device", devices)

# Create input vector with 0s
input_data = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

# Set base values
if 'yj_Amount_spent' in input_data.columns:
    input_data['yj_Amount_spent'] = amount_spent
if 'Amount spent' in input_data.columns:
    input_data['Amount spent'] = amount_spent
input_data['day'] = day
input_data['month'] = month
input_data['weekday'] = weekday

# One-hot encoded inputs
for col in [selected_platform, selected_placement, selected_device]:
    if col in input_data.columns:
        input_data[col] = 1

if 'Ad set budget type_Using ad set budget' in input_data.columns:
    input_data['Ad set budget type_Using ad set budget'] = int(use_ad_set_budget)

# Predict
if st.button("Predict KPIs"):
    try:
        raw_prediction = selected_model.predict(input_data)
        prediction = transformer.inverse_transform(raw_prediction)
        prediction = np.clip(prediction, 0, None)

        # Enforce logical KPI constraints
        if campaign_type == "Awareness":
            reach, impressions = prediction[0]
            if reach > impressions:
                reach = impressions * 0.95  # slightly below impressions
            prediction[0][0], prediction[0][1] = reach, impressions
            
        labels = kpi_labels[campaign_type]
        for i, value in enumerate(prediction[0]):
            st.success(f"{labels[i]}: {value:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")