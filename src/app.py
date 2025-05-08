import pandas as pd
import streamlit as st

from configuration import set_global_styles, load_pipelines, get_user_input, validate_inputs, make_prediction

# Apply global styles
set_global_styles()

# Display hero image
st.image("static/features-bg.jpg",  use_container_width=True)

# Load pipelines
pipelines = load_pipelines()

# App Title
st.markdown("<h1 style='text-align: center; color: #FFD700;'>Fitness Prediction App 🏋️‍♂️♂️</h1>", unsafe_allow_html=True)
st.write("Enter details to predict fitness expenditure.")

# Model Selection
model_option = st.selectbox("Select how many calories you want to burn today, or How much time you have available to workout:", ["Calories to Burn", "Workout Duration"])

# Get user input
user_input, target_feature = get_user_input(model_option)
input_data = pd.DataFrame([user_input])

# Select model based on user choice
model = pipelines["calorie_model_pipeline"] if model_option == "Calories to Burn" else pipelines["time_model_pipeline"]

# Prediction logic
if st.button("Predict"):
    warning_msg = validate_inputs(model_option, user_input)
    if warning_msg:
        st.warning(warning_msg)  # Show warning and stop execution

    else:
        result = make_prediction(model, input_data, target_feature)
        st.success(result)  # Display prediction

# ----- '''Very First Basic Draft''' -----

# with open("../models/pipelines.pkl", "rb") as f:
    # pipelines = pickle.load(f)
# print(pipelines.keys())  # Shows all available keys in the dictionary

# model = pipelines["calorie_model_pipeline"]
# print(type(model))  # Should output <class 'sklearn.pipeline.Pipeline'>

# age = st.number_input("Age", min_value=1, max_value=100)
# gender = st.selectbox("Gender", ["Male", "Female"])
# height = st.number_input("Height (cm)", min_value=100, max_value=250)
# weight = st.number_input("Weight (kg)", min_value=20, max_value=200)
# duration = st.number_input("Workout Duration (mins)", min_value=1, max_value=300)

# input_data = pd.DataFrame({
#     "Age": [age],
#     "Gender": [gender],
#     "Height": [height],
#     "Weight": [weight],
#     "Duration": [duration]
# })