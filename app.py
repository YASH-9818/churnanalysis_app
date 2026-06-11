import os
import json
import warnings
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

warnings.filterwarnings("ignore")

# ── 1. PAGE CONFIGURATION ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ECB Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. PREMIUM CSS STYLING ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght=300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit default headers/footers */
#MainMenu, footer, header { visibility: hidden; }

/* Custom Container Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #334155;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.metric-val {
    font-size: 26px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 4px;
}
.metric-lbl {
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Prediction Output Banners */
.risk-banner-high {
    background-color: #7f1d1d;
    border: 1px solid #f87171;
    padding: 20px;
    border-radius: 8px;
    color: #fca5a5;
    text-align: center;
}
.risk-banner-low {
    background-color: #064e3b;
    border: 1px solid #34d399;
    padding: 20px;
    border-radius: 8px;
    color: #a7f3d0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── 3. DATA & MODEL LOADING FUNCTIONS WITH DUAL MODEL SUPPORT ──────────────
@st.cache_data
def load_raw_data():
    if Path("European_Bank.csv").exists():
        return pd.read_csv("European_Bank.csv")
    columns = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
               "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
    return pd.DataFrame(columns=columns)

def load_selected_model(model_filename):
    base_path = Path("models")
    pkl_path = base_path / model_filename
    json_path = base_path / "feature_cols.json"
    
    # Auto-trigger execution if files are absent on backend startup
    if not pkl_path.exists() or not json_path.exists():
        if Path("train_models.py").exists() and Path("European_Bank.csv").exists():
            subprocess.run(["python", "train_models.py"])
            
    try:
        loaded_model = joblib.load(pkl_path)
        with open(json_path, "r") as f:
            feature_cols = json.load(f)
        return loaded_model, feature_cols
    except Exception as e:
        return None, None

df_raw = load_raw_data()

# ── 4. SIDEBAR GLOBAL FILTER CONTROLS ──────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Filter Engine")
    st.markdown("Segment metrics dynamically across the platform layers.")
    st.markdown("---")
    
    # Model Selection Block
    st.subheader("Model Configuration")
    chosen_architecture = st.selectbox(
        "Active ML Engine",
        options=["Gradient Boosting", "Random Forest"],
        key="sidebar_architecture_select"
    )
    
    model_file_map = {
        "Gradient Boosting": "gradient_boosting.pkl",
        "Random Forest": "random_forest.pkl"
    }
    
    # Dynamic Load Routine Execution
    model, expected_features = load_selected_model(model_file_map[chosen_architecture])
    st.markdown("---")
    
    # Demographic Filtering Blocks
    all_geos = sorted(list(df_raw["Geography"].unique())) if not df_raw.empty else ["France", "Germany", "Spain"]
    selected_geos = st.multiselect("Target Regions", options=all_geos, default=all_geos, key="sidebar_geo_multiselect")
    
    all_genders = sorted(list(df_raw["Gender"].unique())) if not df_raw.empty else ["Male", "Female"]
    selected_genders = st.multiselect("Gender Segments", options=all_genders, default=all_genders, key="sidebar_gender_multiselect")
    
    min_age = int(df_raw["Age"].min()) if not df_raw.empty else 18
    max_age = int(df_raw["Age"].max()) if not df_raw.empty else 100
    selected_age_range = st.slider("Age Parameters", min_age, max_age, (min_age, max_age), key="sidebar_age_slider")

if not df_raw.empty:
    filtered_df = df_raw[
        (df_raw["Geography"].isin(selected_geos)) &
        (df_raw["Gender"].isin(selected_genders)) &
        (df_raw["Age"].between(selected_age_range[0], selected_age_range[1]))
    ]
else:
    filtered_df = df_raw

# ── 5. HEADLINE APP HEADER ──────────────────────────────────────────────────
st.title("📊 Bank Customer Churn Intelligence Platform")
st.markdown(f"An advanced ensemble machine learning system. Operating Engine: **{chosen_architecture}**")
st.markdown("---")

tab_overview, tab_prediction, tab_simulator = st.tabs([
    "📈 Performance Overview", 
    "🔮 Individual Risk Calculator", 
    "🎮 Scenario Simulator"
])

# ── 6. TAB 1: ENTERPRISE PERFORMANCE OVERVIEW ─────────────────────────────
with tab_overview:
    if filtered_df.empty:
        st.warning("⚠️ No records match current selection filters. Adjust the sidebar constraints.")
    else:
        total_cust = len(filtered_df)
        total_exited = int(filtered_df["Exited"].sum())
        churn_rate = (total_exited / total_cust) * 100 if total_cust > 0 else 0
        avg_age = filtered_df["Age"].mean()
        avg_bal = filtered_df["Balance"].mean()
        active_rate = (filtered_df["IsActiveMember"].sum() / total_cust) * 100 if total_cust > 0 else 0
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{total_cust:,}</div><div class="metric-lbl">Total Customers</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ef4444;">{total_exited:,}</div><div class="metric-lbl">Total Exited ({churn_rate:.1f}%)</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_age:.1f} Yrs</div><div class="metric-lbl">Average Age</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">${avg_bal/1000:.1f}K</div><div class="metric-lbl">Avg Balance</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{active_rate:.1f}%</div><div class="metric-lbl">Active Members</div></div>', unsafe_allow_html=True)
            
        st.write("")  
        
        c_left, c_right = st.columns([1.1, 0.9])
        
        with c_left:
            st.markdown("#### Attrition Distribution across Account Balances")
            fig_bal = px.histogram(
                filtered_df, x="Balance", color="Exited", barmode="group",
                color_discrete_map={0: "#334155", 1: "#38bdf8"},
                labels={"Exited": "Status (1=Exited)"}
            )
            fig_bal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_bal, use_container_width=True)
            
            st.markdown("#### Customer Cohorts by Product Density")
            fig_prod = px.histogram(
                filtered_df, x="NumOfProducts", color="Exited", barmode="group",
                color_discrete_map={0: "#475569", 1: "#ef4444"}
            )
            fig_prod.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_prod, use_container_width=True)

        with c_right:
            st.markdown("#### Geographic Market Composition")
            fig_geo = px.pie(
                filtered_df, names="Geography", values="Balance", hole=0.4,
                color_discrete_sequence=["#0284c7", "#0ea5e9", "#38bdf8"]
            )
            fig_geo.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_geo, use_container_width=True)
            
            st.markdown("#### Churn Segment Proportions by Gender")
            fig_gen = px.pie(
                filtered_df, names="Gender", values="Exited", hole=0.5,
                color_discrete_sequence=["#6366f1", "#a5b4fc"]
            )
            fig_gen.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_gen, use_container_width=True)

# ── 7. TAB 2: INDIVIDUAL RISK CALCULATOR ──────────────────────────────────
with tab_prediction:
    st.subheader("Predictive Attrition Pipeline")
    
    if model is None:
        st.error(f"⚠️ The Machine Learning model framework ({chosen_architecture}) could not be loaded. Ensure models/ directory asset maps are cleanly established.")
    else:
        st.markdown(f"Input specific customer parameters below to calculate real-time churn liability scores via **{chosen_architecture}**.")
        
        p1, p2, p3 = st.columns(3)
        with p1:
            inp_geo = st.selectbox("Customer Primary Region", ["France", "Germany", "Spain"], key="calc_geo_select")
            inp_gender = st.selectbox("Customer Gender Identity", ["Male", "Female"], key="calc_gender_select")
            inp_age = st.number_input("Customer Current Age", min_value=18, max_value=100, value=35, key="calc_age_input")
        with p2:
            inp_credit = st.slider("Credit Score Matrix", 300, 850, 650, key="calc_credit_slider")
            inp_tenure = st.slider("Tenure Framework (Years)", 0, 10, 5, key="calc_tenure_slider")
            inp_products = st.number_input("Active Bank Products Used", min_value=1, max_value=4, value=2, key="calc_products_input")
        with p3:
            inp_balance = st.number_input("Current Book Balance ($)", min_value=0.0, value=50000.0, key="calc_balance_input")
            inp_salary = st.number_input("Estimated Household Salary ($)", min_value=0.0, value=75000.0, key="calc_salary_input")
            inp_card = st.checkbox("Active Credit Card Holder", value=True, key="calc_card_check")
            inp_member = st.checkbox("Engaged Active Member Status", value=True, key="calc_member_check")
            
        if st.button("Execute Real-Time Scoring Pipeline", type="primary", key="calc_run_button"):
            raw_input = {
                "CreditScore": inp_credit, "Age": inp_age, "Tenure": inp_tenure,
                "Balance": inp_balance, "NumOfProducts": inp_products,
                "HasCrCard": int(inp_card), "IsActiveMember": int(inp_member),
                "EstimatedSalary": inp_salary, "Geography": inp_geo, "Gender": inp_gender
            }
            input_df = pd.DataFrame([raw_input])
            
            for g in ["France", "Germany", "Spain"]:
                input_df[f"Geography_{g}"] = 1.0 if inp_geo == g else 0.0
            for gen in ["Female", "Male"]:
                input_df[f"Gender_{gen}"] = 1.0 if inp_gender == gen else 0.0
                
            input_df = input_df.drop(columns=["Geography", "Gender"])
            
            input_df["BalanceToSalary"]       = input_df["Balance"] / (input_df["EstimatedSalary"] + 1)
            input_df["ProductDensity"]        = input_df["NumOfProducts"] / (input_df["Tenure"] + 1)
            input_df["EngagementProduct"]     = input_df["IsActiveMember"] * input_df["NumOfProducts"]
            input_df["AgeTenureInteraction"]  = input_df["Age"] * input_df["Tenure"]
            input_df["HighBalance"]           = (input_df["Balance"] > 100000).astype(int)
            input_df["SeniorCustomer"]        = (input_df["Age"] >= 45).astype(int)
            
            input_df = input_df[expected_features]
            
            proba = model.predict_proba(input_df)[0][1]
            risk_pct = proba * 100
            
            st.markdown("---")
            st.subheader("Model Output Summary Matrix")
            
            if risk_pct >= 50.0:
                st.markdown(f"""
                <div class="risk-banner-high">
                    <h3>⚠️ HIGH ATTRITION LIABILITY FORECASTED</h3>
                    <p style="font-size:32px; font-weight:700; margin:10px 0;">Risk Score: {risk_pct:.2f}%</p>
                    <p>This consumer shows severe statistical indicators matching historical attrition portfolios. Actionable mitigation required.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-banner-low">
                    <h3>✅ LOW RISK RETENTION RATING</h3>
                    <p style="font-size:32px; font-weight:700; margin:10px 0;">Risk Score: {risk_pct:.2f}%</p>
                    <p>Account behavior patterns show safe stabilization vectors. Retention probability remains robust.</p>
                </div>
                """, unsafe_allow_html=True)

# ── 8. TAB 3: SCENARIO WHAT-IF SIMULATOR ────────────────────────────────────
with tab_simulator:
    st.subheader("Corporate What-If Scenario Matrix")
    st.markdown(f"Simulate global policy trends using the tracking parameters of: **{chosen_architecture}**")
    
    if model is None or df_raw.empty:
        st.error("⚠️ Ensure dataset and machine learning components are both initialized to unlock the simulator engine.")
    else:
        s1, s2 = st.columns(2)
        with s1:
            mod_salary = st.slider("Global Estimated Salary Shift (%)", -50, 50, 0, step=5, key="sim_salary_slider")
            mod_balance = st.slider("Global Account Balance Adjustment (%)", -50, 50, 0, step=5, key="sim_balance_slider")
        with s2:
            force_active = st.selectbox("Simulate Global Member Engagement Campaign", ["No Change", "Convert All to Active", "Convert All to Inactive"], key="sim_campaign_select")
            
        if st.button("Execute Batch Population Re-Scoring", type="secondary", key="sim_run_button"):
            sim_df = df_raw.copy()
            
            if mod_salary != 0:
                sim_df["EstimatedSalary"] = sim_df["EstimatedSalary"] * (1 + mod_salary/100)
            if mod_balance != 0:
                sim_df["Balance"] = sim_df["Balance"] * (1 + mod_balance/100)
            if force_active == "Convert All to Active":
                sim_df["IsActiveMember"] = 1
            elif force_active == "Convert All to Inactive":
                sim_df["IsActiveMember"] = 0
                
            sim_df = pd.get_dummies(sim_df, columns=["Geography", "Gender"], drop_first=False)
            
            sim_df["BalanceToSalary"]       = sim_df["Balance"] / (sim_df["EstimatedSalary"] + 1)
            sim_df["ProductDensity"]        = sim_df["NumOfProducts"] / (sim_df["Tenure"] + 1)
            sim_df["EngagementProduct"]     = sim_df["IsActiveMember"] * sim_df["NumOfProducts"]
            sim_df["AgeTenureInteraction"]  = sim_df["Age"] * sim_df["Tenure"]
            sim_df["HighBalance"]           = (sim_df["Balance"] > 100000).astype(int)
            sim_df["SeniorCustomer"]        = (sim_df["Age"] >= 45).astype(int)
            
            sim_features = [c for c in expected_features if c in sim_df.columns]
            sim_input = sim_df[sim_features]
            
            sim_probs = model.predict_proba(sim_input)[:, 1]
            sim_preds = (sim_probs >= 0.5).astype(int)
            
            orig_churn = int(df_raw["Exited"].sum())
            new_churn = int(sim_preds.sum())
            variance = new_churn - orig_churn
            
            st.markdown("---")
            st.markdown("#### Simulated Macro Structural Impact Results")
            
            v1, v2, v3 = st.columns(3)
            v1.metric("Baseline Portfolio Attrition Count", f"{orig_churn:,}")
            v2.metric("Simulated Portfolio Attrition Count", f"{new_churn:,}")
            v3.metric("Net Macro Count Variance", f"{variance:+,}", delta_color="inverse")