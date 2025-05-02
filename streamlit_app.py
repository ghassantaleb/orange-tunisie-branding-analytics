import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ------------------------
# MODEL CONFIGURATION
# ------------------------
MODEL_CONFIG = {
    "Traffic": {
        "model_path": "models/rf_ctr_traffic_model.joblib",
        "features_path": "models/traffic_model_features.pkl",
        "target": "CTR (%)"
    },
    "Awareness": {
        "model_path": "models/rf_awareness_reach_impressions_model.joblib",
        "features_path": "models/awareness_model_features.pkl",
        "target": "Reach & Impressions"
    },
    "Engagement": {
        "model_path": "models/xgb_post_engagement_model.joblib",
        "features_path": "models/engagement_model_features.pkl",
        "target": "Post Engagements"
    },
    "Video": {
        "model_path": "models/rf_video_engagement_model.joblib",
        "features_path": "models/video_model_features.pkl",
        "target": "3s Views & Video Plays"
    },
    "App Installs": {
        "model_path": "models/xgb_app_installs_model.joblib",
        "features_path": "models/app_installs_model_features.pkl",
        "target": "App Installs"
    }
}

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="Meta Ads Predictor", layout="centered")

from PIL import Image

logo = Image.open("images/orange_logo.png")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=120)
with col2:
    st.title("Meta Ads KPI Prediction – Orange Tunisie")
st.markdown("""
Select your campaign type and enter inputs to predict key performance indicators. 
The app will show predictions and a visual KPI summary.
""")

# Campaign selection
campaign_type = st.sidebar.selectbox("📂 Select Campaign Type", list(MODEL_CONFIG.keys()))

# Load model and feature names
model_info = MODEL_CONFIG[campaign_type]
model = joblib.load(model_info["model_path"])
features = joblib.load(model_info["features_path"])

# ------------------------
# SHARED INPUTS
# ------------------------
budget = st.number_input("💰 Ad Set Budget (USD)", min_value=1.0, value=50.0)
platform = st.selectbox("📱 Platform", ["Facebook", "Instagram", "Audience Network", "Messenger"])
device = st.selectbox("💻 Device", ["Mobile", "Desktop"])
placement = st.selectbox("📌 Placement", ["Feed", "Stories", "In-Stream", "Search", "Other"])
budget_type = st.selectbox("📅 Budget Type", ["daily", "lifetime"])
day = st.slider("📆 Day of Month", 1, 31, value=15)
month = st.slider("📅 Month", 1, 12, value=4)
weekday = st.slider("📅 Weekday (0=Mon)", 0, 6, value=2)

# ------------------------
# PREP FUNCTION (generic)
# ------------------------
def prepare_features():
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
    df = pd.DataFrame([input_data])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]

# ------------------------
# CAMPAIGN LOGIC & OUTPUTS
# ------------------------
if campaign_type == "Traffic":
    if st.button("🔮 Predict CTR"):
        X_input = prepare_features()
        log_ctr = model.predict(X_input)[0]
        ctr = np.expm1(log_ctr)

        st.success(f"📈 Predicted CTR: **{ctr:.2f}%**")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ctr,
            title={"text": "CTR (%)"},
            gauge={"axis": {"range": [None, 10]},
                   "bar": {"color": "#0072C6"}},
        ))
        st.plotly_chart(fig)

elif campaign_type == "Awareness":
    if st.button("🔮 Predict Reach & Impressions"):
        X_input = prepare_features()
        preds = model.predict(X_input)[0]
        reach = np.expm1(preds[0])
        impressions = np.expm1(preds[1])
        st.success(f"👁️ Predicted Reach: **{reach:,.0f}**")
        st.success(f"📢 Impressions: **{impressions:,.0f}**")

        fig = go.Figure(data=[
            go.Bar(name='Reach', x=['Reach'], y=[reach]),
            go.Bar(name='Impressions', x=['Impressions'], y=[impressions])
        ])
        fig.update_layout(barmode='group', title="Reach vs Impressions")
        st.plotly_chart(fig)

elif campaign_type == "Engagement":
 if st.button("🔮 Predict Post Engagements"):
        X_input = prepare_features()
        log_eng = model.predict(X_input)[0]
        engagements = np.expm1(log_eng)

        st.success(f"💬 Predicted Post Engagements: **{engagements:,.0f}**")
        st.bar_chart(pd.DataFrame({"Post Engagements": [engagements]}))
elif campaign_type == "Video":
 if st.button("🔮 Predict Video KPIs"):
        X_input = prepare_features()
        preds = model.predict(X_input)[0]
        views_3s = np.expm1(preds[0])
        full_plays = np.expm1(preds[1])
        st.success(f"🎬 3s Video Plays: **{views_3s:,.0f}**")
        st.success(f"▶️ Full Video Plays: **{full_plays:,.0f}**")

        st.bar_chart(pd.DataFrame({"3s Views": [views_3s], "Full Plays": [full_plays]}))
elif campaign_type == "App Installs":
   if st.button("🔮 Predict App Installs"):
        X_input = prepare_features()
        log_installs = model.predict(X_input)[0]
        installs = np.expm1(log_installs)

        st.success(f"📱 Predicted App Installs: **{installs:,.0f}**")
        st.bar_chart(pd.DataFrame({"App Installs": [installs]}))
