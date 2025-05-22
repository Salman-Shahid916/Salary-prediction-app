import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load saved model and preprocessors
model = tf.keras.models.load_model("salary_model.keras")

with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('onehot_encoder_job.pkl', 'rb') as f:
    job_encoder = pickle.load(f)

with open('onehot_encoder_edu.pkl', 'rb') as f:
    edu_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Salary Prediction App")

# Input fields
gender_input = st.selectbox("Gender", gender_encoder.classes_)
education_input = st.selectbox("Education", edu_encoder.categories_[0])
job_input = st.selectbox("Job Title", job_encoder.categories_[0])
experience_input = st.slider("Years of Experience", 0, 40, 5)
age_input = st.slider("Age", 20, 60, 30)

# Preprocess inputs
gender_encoded = gender_encoder.transform([gender_input])[0]  # label encoded => single number
edu_encoded = edu_encoder.transform([[education_input]]).toarray()
job_encoded = job_encoder.transform([[job_input]]).toarray()

numerical_data = np.array([[gender_encoded, experience_input, age_input]])
input_data = np.concatenate([numerical_data, job_encoded, edu_encoded], axis=1)

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
predicted_salary = model.predict(scaled_input)[0][0]

# Display result
st.subheader(f"Estimated Salary: ${predicted_salary:,.2f}")
