import streamlit as st
import pandas as pd
import joblib

# Load the model and encoder
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Salary prediction app")

# User inputs
age = st.number_input("Age", 18, 60)
gender = st.selectbox("Gender", encoder["Gender"].classes_)
education = st.selectbox("Education Level", encoder["Education Level"].classes_)
job_title = st.selectbox("Job Title", encoder["Job Title"].classes_)
year_of_exp = st.number_input("Year of Experience", 0, 40)

# Create DataFrame for prediction
df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "YearsExperience": [year_of_exp]
})

# Prediction logic
if st.button("Predict"):
    # Everything inside the 'if' must be indented
    for col in encoder:
        # Everything inside the 'for' must be indented further
        df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)
    st.success(f"Predicted Salary: {prediction[0]:,.2f}")
