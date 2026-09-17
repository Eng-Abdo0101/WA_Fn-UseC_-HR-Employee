import streamlit as st
import pandas as pd
import joblib


attrition_model = joblib.load("employee_attrition_catboost_model.pkl")
attrition_features = joblib.load("employee_features.pkl")

salary_model = joblib.load("employee_salary_xgb_model.pkl")
salary_scaler = joblib.load("employee_salary_scaler.pkl")
salary_features = joblib.load("employee_salary_features.pkl")


st.set_page_config(
    page_title="Employee Attrition & Salary Prediction",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme toggle (dark / light) — injects CSS variables and component styling
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("Dark mode", value=True)
    st.divider()
    st.caption(
        "Fill in an employee's profile once — both the attrition risk and "
        "salary estimate use the same shared fields."
    )

if dark_mode:
    bg, card, text, subtext, accent, accent2, border = (
        "#0e1117", "#161b22", "#f0f2f6", "#9aa4b2", "#6ee7b7", "#60a5fa", "#262d3a"
    )
else:
    bg, card, text, subtext, accent, accent2, border = (
        "#f7f8fa", "#ffffff", "#111318", "#5b6472", "#059669", "#2563eb", "#e6e8ec"
    )

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {card};
        border-right: 1px solid {border};
    }}
    div[data-testid="stMetric"] {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {subtext} !important; }}
    .card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }}
    .hero {{
        background: linear-gradient(135deg, {accent2}22, {accent}22);
        border: 1px solid {border};
        border-radius: 20px;
        padding: 28px 30px;
        margin-bottom: 22px;
    }}
    .hero h1 {{ margin: 0 0 6px 0; font-size: 1.9rem; }}
    .hero p {{ margin: 0; color: {subtext}; }}
    .pill {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }}
    .pill-risk {{ background: #ef444422; color: #ef4444; }}
    .pill-safe {{ background: {accent}22; color: {accent}; }}
    .gauge-track {{
        width: 100%; height: 14px; border-radius: 999px;
        background: linear-gradient(90deg, {accent} 0%, #facc15 50%, #ef4444 100%);
        position: relative; margin: 10px 0 4px 0;
    }}
    .gauge-marker {{
        position: absolute; top: -6px; width: 4px; height: 26px;
        background: {text}; border-radius: 2px;
    }}
    .stButton>button {{
        border-radius: 10px; font-weight: 600; padding: 0.5rem 1.2rem;
    }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    hr {{ border-color: {border}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>👨‍💼 Employee Attrition &amp; Salary Prediction</h1>
        <p>Estimate flight risk and fair compensation from a single employee profile — powered by CatBoost &amp; XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared input form
# ---------------------------------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Employee Profile")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Personal**")
    age = st.number_input("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    distance = st.number_input("Distance From Home", 1, 29, 5)
    education = st.selectbox(
        "Education Level", [1, 2, 3, 4, 5], index=2,
        format_func=lambda x: {
            1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor",
            4: "4 - Master", 5: "5 - Doctor",
        }[x],
    )
    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
    )

with c2:
    st.markdown("**Job**")
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ],
    )
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    num_companies = st.number_input("Num Companies Worked", 0, 9, 2)

with c3:
    st.markdown("**Compensation**")
    daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
    hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
    monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
    percent_salary_hike = st.slider("Percent Salary Hike (last review)", 11, 25, 15)
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Satisfaction**")
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
with c2:
    st.markdown("**Engagement**")
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
    performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4], index=2)
with c3:
    st.markdown("**Tenure**")
    total_working_years = st.number_input("Total Working Years", 0, 40, 8)
    years_company = st.number_input("Years At Company", 0, 40, 5)
    years_in_role = st.number_input("Years In Current Role", 0, 18, 3)
    years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    years_with_manager = st.number_input("Years With Current Manager", 0, 17, 3)
    training_times = st.number_input("Training Times Last Year", 0, 6, 2)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MonthlyIncome / Attrition linkage
#
# MonthlyIncome only feeds the attrition model; the salary model predicts
# MonthlyIncome, so it can't also take it as an input. To keep the two tabs
# connected instead of feeling like separate tools:
#   - "Use predicted salary for attrition check" lets the salary estimate
#     (once generated) feed straight into the attrition model, instead of
#     typing income twice.
#   - Otherwise the user enters income manually for the attrition tab.
# ---------------------------------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Income Handling")
link_income = st.toggle(
    "Link Salary Estimate → Attrition Check (use predicted salary instead of typing it)",
    value=False,
)
manual_income = st.number_input(
    "Monthly Income (used for attrition check)",
    1000, 20000, 5000,
    disabled=link_income,
    help="Disabled when linking to the predicted salary above.",
)
st.markdown("</div>", unsafe_allow_html=True)

if "predicted_salary" not in st.session_state:
    st.session_state.predicted_salary = None


def build_engineered_row(raw: dict) -> pd.DataFrame:
    """Recreate the exact engineered features from the notebook (cells 26-32)."""
    df = pd.DataFrame([raw])

    def age_group(a):
        if a < 30:
            return "Young"
        elif a < 45:
            return "Adult"
        elif a < 60:
            return "Middle_Aged"
        else:
            return "Senior"

    df["AgeGroup"] = df["Age"].apply(age_group)
    df["OverallSatisfaction"] = (
        df["JobSatisfaction"] + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"] + df["JobInvolvement"]
    )
    df["CompanyTenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["PromotionDelayRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["OverTime_WorkLife"] = (df["OverTime"] == "Yes").astype(int) * df["WorkLifeBalance"]
    return df


shared_raw = {
    "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance, "Education": education,
    "EnvironmentSatisfaction": environment_satisfaction, "HourlyRate": hourly_rate,
    "JobInvolvement": job_involvement, "JobLevel": job_level, "JobSatisfaction": job_satisfaction,
    "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies,
    "PercentSalaryHike": percent_salary_hike, "PerformanceRating": performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction, "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": training_times,
    "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_company,
    "YearsInCurrentRole": years_in_role, "YearsSinceLastPromotion": years_since_promotion,
    "YearsWithCurrManager": years_with_manager, "BusinessTravel": business_travel,
    "Department": department, "EducationField": education_field, "Gender": gender,
    "JobRole": job_role, "MaritalStatus": marital_status, "OverTime": overtime,
}

tab1, tab2 = st.tabs(["🚪  Attrition Risk", "💰  Salary Estimate"])

# ---------------------------------------------------------------------------
# Salary tab
# ---------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button("Predict Salary", type="primary"):
        input_df = build_engineered_row(shared_raw)
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_aligned = input_encoded.reindex(columns=salary_features, fill_value=0)
        input_scaled = salary_scaler.transform(input_aligned)

        predicted_salary = float(salary_model.predict(input_scaled)[0])
        st.session_state.predicted_salary = predicted_salary

    if st.session_state.predicted_salary is not None:
        predicted_salary = st.session_state.predicted_salary
        delta = manual_income - predicted_salary

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Fair Salary", f"${predicted_salary:,.0f}")
        m2.metric("Entered Salary", f"${manual_income:,.0f}")
        m3.metric(
            "Gap vs. Estimate",
            f"${abs(delta):,.0f}",
            delta=f"{'Overpaid' if delta > 0 else 'Underpaid'}" if abs(delta) > 200 else "In line",
            delta_color="inverse" if delta > 0 else "normal",
        )

        if delta < -500:
            st.info("Entered salary is notably below the model's estimate — a retention risk factor if this reflects reality.")
        elif delta > 500:
            st.info("Entered salary is above the model's estimate for this profile.")
        else:
            st.success("Entered salary is close to the model's estimate for this profile.")

        with st.expander("📊 Feature contributions (model inputs)"):
            input_df = build_engineered_row(shared_raw)
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            input_aligned = input_encoded.reindex(columns=salary_features, fill_value=0)
            st.dataframe(input_aligned.T.rename(columns={0: "value"}), use_container_width=True)
    else:
        st.caption("Click **Predict Salary** to generate an estimate.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Attrition tab
# ---------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    threshold = st.slider(
        "Decision threshold (lower = flags more employees as at-risk)",
        0.1, 0.9, 0.30, 0.05,
    )

    income_for_check = (
        st.session_state.predicted_salary
        if (link_income and st.session_state.predicted_salary is not None)
        else manual_income
    )
    if link_income and st.session_state.predicted_salary is None:
        st.warning("Linking is on, but no salary has been predicted yet — generate one in the Salary tab first. Using entered income for now.")
        income_for_check = manual_income

    st.caption(f"Using Monthly Income: **${income_for_check:,.0f}** "
               f"({'linked from salary estimate' if link_income and st.session_state.predicted_salary else 'manually entered'})")

    if st.button("Predict Attrition", type="primary"):
        raw = dict(shared_raw)
        raw["MonthlyIncome"] = income_for_check

        input_df = build_engineered_row(raw)
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_final = input_encoded.reindex(columns=attrition_features, fill_value=0)

        probability = float(attrition_model.predict_proba(input_final)[0][1])
        prediction = int(probability >= threshold)

        marker_pos = min(max(probability * 100, 0), 100)
        label = "AT RISK" if prediction == 1 else "LIKELY TO STAY"
        pill_class = "pill-risk" if prediction == 1 else "pill-safe"

        st.markdown(f'<span class="pill {pill_class}">{label}</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="gauge-track">
                <div class="gauge-marker" style="left: calc({marker_pos}% - 2px);"></div>
            </div>
            <p style="font-size:0.8rem; color:{subtext}; display:flex; justify-content:space-between;">
                <span>Low risk</span><span>High risk</span>
            </p>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Attrition Probability", f"{probability:.1%}")
        m2.metric("Decision Threshold", f"{threshold:.0%}")
        m3.metric("Predicted Outcome", "Leave" if prediction else "Stay")

        with st.expander("📊 Model inputs used"):
            st.dataframe(input_final.T.rename(columns={0: "value"}), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Attrition model: CatBoost (class-weighted) · Salary model: tuned XGBoost · Built with Streamlit")
