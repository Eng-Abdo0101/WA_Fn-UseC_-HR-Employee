import streamlit as st
import pandas as pd
import joblib


model = joblib.load("employee_attrition_catboost_model.pkl")
features = joblib.load("employee_features.pkl")


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("Employee Attrition Prediction")
st.write(
    "Predict whether an employee is likely to leave the company using Machine Learning."
)

# ---------------------------------------------------------------------------
# The model was trained on 51 features (raw columns + engineered features),
# not just a handful of inputs. To get an accurate prediction we need to
# collect (almost) everything the training pipeline used, then rebuild the
# exact same engineered features before encoding.
# ---------------------------------------------------------------------------

st.header("Personal Info")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
with c2:
    distance = st.number_input("Distance From Home", 1, 29, 5)
    education = st.selectbox(
        "Education Level",
        [1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor",
            4: "4 - Master", 5: "5 - Doctor"
        }[x]
    )
    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
    )

st.header("Job Info")
c1, c2 = st.columns(2)
with c1:
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ]
    )
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
with c2:
    business_travel = st.selectbox(
        "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    num_companies = st.number_input("Num Companies Worked", 0, 9, 2)

st.header("Compensation")
c1, c2 = st.columns(2)
with c1:
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
with c2:
    hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
    monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
percent_salary_hike = st.slider("Percent Salary Hike (last review)", 11, 25, 15)
stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])

st.header("Satisfaction & Engagement")
c1, c2 = st.columns(2)
with c1:
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
with c2:
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4], index=2)

st.header("Work History")
c1, c2 = st.columns(2)
with c1:
    total_working_years = st.number_input("Total Working Years", 0, 40, 8)
    years_company = st.number_input("Years At Company", 0, 40, 5)
    years_in_role = st.number_input("Years In Current Role", 0, 18, 3)
with c2:
    years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    years_with_manager = st.number_input("Years With Current Manager", 0, 17, 3)
    training_times = st.number_input("Training Times Last Year", 0, 6, 2)

# This is the empirically-best decision threshold found in the notebook's
# threshold-tuning step (cell 112), not the default 0.5 — with class weights
# skewed toward the minority class, 0.5 under-predicts attrition.
threshold = st.slider(
    "Decision threshold (lower = flags more employees as at-risk)",
    0.1, 0.9, 0.30, 0.05
)

if st.button("Predict"):

    raw = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime,
    }

    input_df = pd.DataFrame([raw])

    # --- Recreate the exact engineered features from the notebook ---
    def age_group(a):
        if a < 30:
            return "Young"
        elif a < 45:
            return "Adult"
        elif a < 60:
            return "Middle_Aged"
        else:
            return "Senior"

    input_df["AgeGroup"] = input_df["Age"].apply(age_group)

    input_df["OverallSatisfaction"] = (
        input_df["JobSatisfaction"]
        + input_df["EnvironmentSatisfaction"]
        + input_df["RelationshipSatisfaction"]
        + input_df["JobInvolvement"]
    )

    input_df["CompanyTenureRatio"] = (
        input_df["YearsAtCompany"] / (input_df["TotalWorkingYears"] + 1)
    )

    input_df["PromotionDelayRatio"] = (
        input_df["YearsSinceLastPromotion"] / (input_df["YearsAtCompany"] + 1)
    )

    input_df["OverTime_WorkLife"] = (
        (input_df["OverTime"] == "Yes").astype(int) * input_df["WorkLifeBalance"]
    )

    # --- One-hot encode categoricals the same way as training ---
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # --- Align to the exact columns the model was trained on ---
    input_final = input_encoded.reindex(columns=features, fill_value=0)

    probability = model.predict_proba(input_final)[0][1]
    prediction = int(probability >= threshold)

    if prediction == 1:
        st.error(f"Employee is likely to leave the company — Probability: {probability:.2f}")
    else:
        st.success(f"Employee is likely to stay — Probability: {probability:.2f}")

    with st.expander("Show model inputs"):
        st.dataframe(input_final)
