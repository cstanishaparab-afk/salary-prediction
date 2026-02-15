import streamlit as st
import pandas as pd
import joblib

# These lines are where the error is happening
# It means "salary_prediction_model.pkl" is likely empty or 0 bytes on GitHub
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Salary prediction app")

age = st.number_input("Age", 18, 60)
gender = st.selectbox("Gender", encoder["Gender"].classes_)
education = st.selectbox("Education Level", encoder["Education Level"].classes_)
job_title = st.selectbox("Job Title", encoder["Job Title"].classes_)
year_of_exp = st.number_input("Year of Experience", 0, 40)

df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "YearsExperience": [year_of_exp]
})

if st.button("Predict"):
    # 1. First, loop through and encode all columns
    for col in encoder:
        df[col] = encoder[col].transform(df[col])
        
    # 2. NOW (outside the for-loop but inside the if-block), run the prediction
   prediction = model.predict(df)
   st.success(f"Predicted Salary: {prediction[0]:,.2f}")
