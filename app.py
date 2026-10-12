from c45_model import C45DecisionTree
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Grade Prediction")
st.write("Predict a student's grade using the C4.5 Decision Tree.")

# Load the saved model
@st.cache_resource
def load_model():
    with open("c45_grade_model.pkl", "rb") as file:
        return pickle.load(file)

model_data = load_model()
model = model_data["model"]
FEATURES = model_data["features"]

st.subheader("Enter Student Details")

with st.form("student_form"):
    attendance = st.number_input(
        "Attendance (%)", min_value=0.0, max_value=100.0, value=90.0
    )
    midterm = st.number_input(
        "Midterm Score", min_value=0.0, max_value=100.0, value=75.0
    )
    assignments = st.number_input(
        "Assignments Average", min_value=0.0, max_value=100.0, value=80.0
    )
    quizzes = st.number_input(
        "Quizzes Average", min_value=0.0, max_value=100.0, value=78.0
    )
    participation = st.number_input(
        "Participation Score", min_value=0.0, max_value=100.0, value=75.0
    )
    projects = st.number_input(
        "Projects Score", min_value=0.0, max_value=100.0, value=82.0
    )
    study_hours = st.number_input(
        "Study Hours per Week", min_value=0.0, max_value=168.0, value=15.0
    )
    sleep_hours = st.number_input(
        "Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.5
    )

    submitted = st.form_submit_button("Predict Grade")

if submitted:
    student = pd.DataFrame([{
        "Attendance (%)": attendance,
        "Midterm_Score": midterm,
        "Assignments_Avg": assignments,
        "Quizzes_Avg": quizzes,
        "Participation_Score": participation,
        "Projects_Score": projects,
        "Study_Hours_per_Week": study_hours,
        "Sleep_Hours_per_Night": sleep_hours
    }], columns=FEATURES)

    prediction = model.predict(student)[0]

    st.success(f"Predicted Grade: {prediction}")
    st.caption(
        "This is a model prediction, not a guarantee of the student's actual grade."
    )

    with st.expander("View submitted inputs"):
        st.dataframe(student, use_container_width=True)
