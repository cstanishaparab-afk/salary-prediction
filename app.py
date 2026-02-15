import streamlit as st
import pandas as pd
import joblib

# These lines run as soon as the app starts
# If the error is on line 6 or 7, your .pkl files are the problem
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
    # This loop handles the encoding transformation
    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)
    st.success(f"Predicted Salary: {prediction[0]:,.2f}")
