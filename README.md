#  Meta Ads Predictive Modeling – Orange Tunisie

##  Project Overview

This project builds a full end-to-end predictive modeling suite for Orange Tunisie's Meta Ads campaigns.  
The goal is to forecast key KPIs for various campaign objectives (Traffic, Engagement, Video Views, Awareness, App Installs) and optimize future marketing budgets.

Unlike e-commerce setups, Orange's campaigns are branding-focused, requiring specialized predictive approaches.

---

##  Objectives

- Predict key KPIs (CTR, Engagement, Reach, Impressions, Installs)
- Build models adapted to branding campaigns (without ROAS/purchases)
- Prepare models for real-time dashboards and deployment
- Highlight technical challenges, strengths, and future opportunities

---

##  Techniques and Tools Used

| Task                   | Tool/Technique |
|-------------------------|----------------|
| Data Cleaning           | Pandas, manual Excel handling |
| Feature Engineering     | Time Features, Log Transformations |
| Encoding Categorical Data | One-Hot Encoding |
| Modeling Techniques     | Linear Regression, Ridge Regression, Random Forest, XGBoost |
| Multi-Output Prediction | MultiOutputRegressor |
| Evaluation Metrics      | RMSE, R² Score |
| Model Saving            | `joblib` |
| Version Control         | Git, GitHub |
| Visualization           | Seaborn, Matplotlib |

---

##  Data Preparation

- Handled missing values manually
- Partitioned data based on campaign objectives
- Performed feature engineering (log transforms, time extraction)
- One-hot encoded categorical variables
- Selected final features based on EDA and correlation analysis

---

##  Modeling Results

| Campaign Type | Target KPI(s) | Best Model    | RMSE | R² Score | Notes |
|---------------|---------------|---------------|------|----------|---------------|
| Traffic       | CTR            | Random Forest | 0.2688 | 0.7386 | Good performance |
| Engagement    | Post Engagement | Tuned XGBoost | 0.8101 | 0.6821 | Reasonable |
| Video Views   | 3-Second Plays, Video Plays | Random Forest | 0.1207–0.1381 | 0.9912–0.9892 | Very good |
| Awareness     | Reach, Impressions | Random Forest | 0.1166–0.2147 | 0.9648–0.9855 | Outstanding |
| App Installs  | App Installs | XGBoost | 0.7490 | 0.8946 | Good, but data limited |


---

##  Strengths of the Project

- Full professional end-to-end pipeline (cleaning → modeling → saving)
- Excellent modeling performance on Traffic, Video, Awareness
- Modular and scalable structure
- Realistic handling of branding-focused campaign goals

---

##  Challenges and Limitations

- Some campaigns (especially App Installs) had limited data (less than 20 rows), which weakens robustness despite good metrics.
- Model performance variability: while most models performed well (especially Awareness), some models like Engagement showed moderate R² (~0.68) due to data noise.
- No ROAS/conversion metrics: models were adapted to branding KPIs only (CTR, Reach, Engagement).

---
## Conclusion
- Despite facing typical real-world marketing dataset issues (missing data, low data for some objectives, noisy variables),
- This project successfully built a robust, scalable, and deployable predictive modeling suite.

- The models are ready to be used in strategic marketing planning, real-time dashboard monitoring, and budget optimization.

- With future data enrichment and deployment integration, this pipeline could power real-world decision-making at Orange Tunisie for branding campaigns across Meta platforms.

##  Future Directions

- Improve App Installs model by collecting more data
- Integrate external factors (holidays, seasonality)
- Hyperparameter optimization (GridSearchCV, Optuna)
- Deploy models into real-time dashboards (Power BI / Streamlit)
- Expose predictive models via APIs (FastAPI, Flask)
- Extend to Google Ads data

---

##  Project Structure
📁 data/ 
    └── Cleaned Meta Ads export 
📁 models/ 
    └── Saved best models (.joblib) 
📁 notebooks/ 
    └── EDA and modeling notebooks 
  📄 README.md 
  📄 .gitignore

##  Author

- **Project Owner**: Ghassen Taleb
- **Date**: April 2025

---

##  License

This project is licensed under the MIT License.

