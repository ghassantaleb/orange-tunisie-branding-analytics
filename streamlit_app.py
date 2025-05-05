import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and features
@st.cache_resource
def load_model_and_features():
    models = {
        'Awareness': joblib.load('C:/Users/ghass/orange-tunisie-branding-analytics/notebooks/models/random_forest_awareness_model.joblib'),
        'Engagement': joblib.load('C:/Users/ghass/orange-tunisie-branding-analytics/notebooks/models/random_forest_engagement_model.joblib'),
        'Traffic': joblib.load('C:/Users/ghass/orange-tunisie-branding-analytics/notebooks/models/bayesian_ridge_traffic_model.joblib')
    }
    features = {
        'Awareness': joblib.load('C:/Users/ghass/orange-tunisie-branding-analytics/models/awareness_model_features.pkl'),
        'Engagement': joblib.load('C:/Users/ghass/orange-tunisie-branding-analytics/models/engagement_model_features.pkl'),
        'Traffic': joblib.load('C:/Users/ghass/orange-tunisie-branding-analytics/models/traffic_model_features.pkl')
    }
    return models, features

models, features = load_model_and_features()

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

# Input placeholders
st.header("Campaign Design Inputs")

# Time Features
day = st.slider("Day of Month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
weekday = st.selectbox("Weekday", list(range(7)))  # 0 = Monday

# Platform Options
platforms = ['Platforme_Facebook', 'Platforme_Instagram']
selected_platform = st.selectbox("Platform", platforms)

# Placement Options
placements = [col for col in model_features if 'Placement_' in col]
selected_placement = st.selectbox("Placement", placements)

# Device Options
devices = [col for col in model_features if 'Device_' in col]
selected_device = st.selectbox("Device", devices)

# Budget Type
budget_type = st.checkbox("Use Ad Set Budget?")

# Create input vector with 0s
input_data = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
input_data['day'] = day
input_data['month'] = month
input_data['weekday'] = weekday

# One-hot encoded inputs
for col in [selected_platform, selected_placement, selected_device]:
    if col in input_data.columns:
        input_data[col] = 1

if 'Ad set budget type_Using ad set budget' in input_data.columns:
    input_data['Ad set budget type_Using ad set budget'] = int(budget_type)

# Predict
if st.button("Predict KPIs"):
    prediction = selected_model.predict(input_data)
    labels = kpi_labels[campaign_type]
    if prediction.shape[1] == 1:
        st.success(f"{labels[0]}: {prediction[0, 0]:.2f}")
    else:
        for i, value in enumerate(prediction[0]):
            st.success(f"{labels[i]}: {value:.2f}")
