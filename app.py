import streamlit as st
import pandas as pd
import joblib


model = joblib.load(
    "employee_attrition_catboost_model.pkl"
)

features = joblib.load(
    "employee_features.pkl"
)


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)


st.title("Employee Attrition Prediction")

st.write(
    "Predict whether an employee is likely to leave the company using Machine Learning."
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)


monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=20000,
    value=5000
)


distance = st.number_input(
    "Distance From Home",
    min_value=1,
    max_value=30,
    value=5
)


years_company = st.number_input(
    "Years At Company",
    min_value=0,
    max_value=40,
    value=5
)


job_satisfaction = st.selectbox(
    "Job Satisfaction",
    [1,2,3,4]
)


work_life = st.selectbox(
    "Work Life Balance",
    [1,2,3,4]
)


overtime = st.selectbox(
    "OverTime",
    ["Yes","No"]
)


job_role = st.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager"
    ]
)


business_travel = st.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)


if st.button("Predict"):


    input_data = pd.DataFrame({

        "Age":[age],

        "MonthlyIncome":[monthly_income],

        "DistanceFromHome":[distance],

        "YearsAtCompany":[years_company],

        "JobSatisfaction":[job_satisfaction],

        "WorkLifeBalance":[work_life],

        "OverTime_"+overtime:[1],

        "JobRole_"+job_role:[1],

        "BusinessTravel_"+business_travel:[1]

    })


    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )


    prediction = model.predict(
        input_data
    )


    probability = model.predict_proba(
        input_data
    )[0][1]


    if prediction[0] == 1:

        st.error(
            f"Employee is likely to leave the company\nProbability: {probability:.2f}"
        )

    else:

        st.success(
            f"Employee is likely to stay\nProbability: {probability:.2f}"
        )