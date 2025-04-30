import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
        "model_path": "models/rf_video_model.joblib",
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

st.title("📊 Meta Ads KPI Prediction – Orange Tunisie")
st.markdown("Select your campaign type and enter inputs to predict key performance indicators.")

# Campaign selection
campaign_type = st.selectbox("📂 Select Campaign Type", list(MODEL_CONFIG.keys()))
st.markdown(f"You selected: **{campaign_type}**")

# Load model and feature names
model_info = MODEL_CONFIG[campaign_type]
model = joblib.load(model_info["model_path"])
features = joblib.load(model_info["features_path"])

# ------------------------
# TRAFFIC CAMPAIGN INPUTS
# ------------------------
if campaign_type == "Traffic":
    st.header("📥 Enter Traffic Campaign Settings")

    budget = st.number_input("Ad Set Budget (USD)", min_value=1.0, value=50.0)
    platform = st.selectbox("Platform", ["Facebook", "Instagram", "Audience Network", "Messenger"])
    device = st.selectbox("Device", ["Mobile", "Desktop"])
    placement = st.selectbox("Placement", ["Feed", "Stories", "In-Stream", "Search", "Other"])
    budget_type = st.selectbox("Budget Type", ["daily", "lifetime"])
    day = st.slider("Day of Month", 1, 31, value=15)
    month = st.slider("Month", 1, 12, value=4)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, value=2)

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

    if st.button("🔮 Predict CTR"):
        X_input = prepare_features()
        log_ctr = model.predict(X_input)[0]
        ctr = np.expm1(log_ctr)
        st.success(f"📈 Predicted CTR: **{ctr:.2f}%**")

# ------------------------
# AWARENESS CAMPAIGN INPUTS
# ------------------------
elif campaign_type == "Awareness":
    st.header("📥 Enter Awareness Campaign Settings")

    budget = st.number_input("Ad Set Budget (USD)", min_value=1.0, value=60.0)
    platform = st.selectbox("Platform", ["Facebook", "Instagram"])
    placement = st.selectbox("Placement", ["Feed", "Stories"])
    device = st.selectbox("Device", ["Mobile", "Desktop"])
    budget_type = st.selectbox("Budget Type", ["daily", "lifetime"])
    day = st.slider("Day of Month", 1, 31, value=15)
    month = st.slider("Month", 1, 12, value=4)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, value=2)

    def prepare_features():
        input_data = {
            "log_spend": np.log1p(budget),
            "day": day,
            "month": month,
            "weekday": weekday,
            "Platforme_Instagram": int(platform == "Instagram"),
            "Placement_Stories": int(placement == "Stories"),
            "Device_Mobile": int(device == "Mobile"),
            "Ad set budget type_lifetime": int(budget_type == "lifetime")
        }
        df = pd.DataFrame([input_data])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        return df[features]

    if st.button("🔮 Predict Reach & Impressions"):
        X_input = prepare_features()
        preds = model.predict(X_input)[0]
        reach = np.expm1(preds[0])
        impressions = np.expm1(preds[1])
        st.success(f"👁️ Predicted Reach: **{reach:,.0f}**\n\n📢 Impressions: **{impressions:,.0f}**")

# ------------------------
# ENGAGEMENT CAMPAIGN INPUTS
# ------------------------
elif campaign_type == "Engagement":
    st.header("📥 Enter Engagement Campaign Settings")

    budget = st.number_input("Ad Set Budget (USD)", min_value=1.0, value=40.0)
    platform = st.selectbox("Platform", ["Facebook", "Instagram"])
    placement = st.selectbox("Placement", ["Feed", "Stories"])
    device = st.selectbox("Device", ["Mobile", "Desktop"])
    budget_type = st.selectbox("Budget Type", ["daily", "lifetime"])
    day = st.slider("Day of Month", 1, 31, value=15)
    month = st.slider("Month", 1, 12, value=4)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, value=2)

    def prepare_features():
        input_data = {
            "log_spend": np.log1p(budget),
            "day": day,
            "month": month,
            "weekday": weekday,
            "Platforme_Instagram": int(platform == "Instagram"),
            "Placement_Stories": int(placement == "Stories"),
            "Device_Mobile": int(device == "Mobile"),
            "Ad set budget type_lifetime": int(budget_type == "lifetime")
        }
        df = pd.DataFrame([input_data])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        return df[features]

    if st.button("🔮 Predict Post Engagements"):
        X_input = prepare_features()
        log_eng = model.predict(X_input)[0]
        engagements = np.expm1(log_eng)
        st.success(f"💬 Predicted Post Engagements: **{engagements:,.0f}**")

# ------------------------
# VIDEO CAMPAIGN INPUTS
# ------------------------
elif campaign_type == "Video":
    st.header("📥 Enter Video Campaign Settings")

    budget = st.number_input("Ad Set Budget (USD)", min_value=1.0, value=55.0)
    platform = st.selectbox("Platform", ["Facebook", "Instagram"])
    placement = st.selectbox("Placement", ["Feed", "Stories"])
    device = st.selectbox("Device", ["Mobile", "Desktop"])
    budget_type = st.selectbox("Budget Type", ["daily", "lifetime"])
    day = st.slider("Day of Month", 1, 31, value=15)
    month = st.slider("Month", 1, 12, value=4)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, value=2)

    def prepare_features():
        input_data = {
            "log_spend": np.log1p(budget),
            "day": day,
            "month": month,
            "weekday": weekday,
            "Platforme_Instagram": int(platform == "Instagram"),
            "Placement_Stories": int(placement == "Stories"),
            "Device_Mobile": int(device == "Mobile"),
            "Ad set budget type_lifetime": int(budget_type == "lifetime")
        }
        df = pd.DataFrame([input_data])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        return df[features]

    if st.button("🔮 Predict Video KPIs"):
        X_input = prepare_features()
        preds = model.predict(X_input)[0]
        views_3s = np.expm1(preds[0])
        full_plays = np.expm1(preds[1])
        st.success(f"🎬 3s Video Plays: **{views_3s:,.0f}**\n▶️ Full Video Plays: **{full_plays:,.0f}**")

# ------------------------
# APP INSTALLS CAMPAIGN INPUTS
# ------------------------
elif campaign_type == "App Installs":
    st.header("📥 Enter App Install Campaign Settings")

    budget = st.number_input("Ad Set Budget (USD)", min_value=1.0, value=70.0)
    platform = st.selectbox("Platform", ["Facebook", "Instagram"])
    placement = st.selectbox("Placement", ["Feed", "Stories"])
    device = st.selectbox("Device", ["Mobile", "Desktop"])
    budget_type = st.selectbox("Budget Type", ["daily", "lifetime"])
    day = st.slider("Day of Month", 1, 31, value=15)
    month = st.slider("Month", 1, 12, value=4)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, value=2)

    def prepare_features():
        input_data = {
            "log_spend": np.log1p(budget),
            "day": day,
            "month": month,
            "weekday": weekday,
            "Platforme_Instagram": int(platform == "Instagram"),
            "Placement_Stories": int(placement == "Stories"),
            "Device_Mobile": int(device == "Mobile"),
            "Ad set budget type_lifetime": int(budget_type == "lifetime")
        }
        df = pd.DataFrame([input_data])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        return df[features]

    if st.button("🔮 Predict App Installs"):
        X_input = prepare_features()
        log_installs = model.predict(X_input)[0]
        installs = np.expm1(log_installs)
        st.success(f"📱 Predicted App Installs: **{installs:,.0f}**")
