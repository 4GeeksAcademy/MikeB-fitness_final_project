import pickle
import pandas as pd
import streamlit as st

def set_global_styles():
    """Applies global styles for the app."""
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(to bottom, #0F2027, #203A43, #2C5364); }
        .stButton>button { background-color: #32CD32; color: white; font-size: 16px; border-radius: 10px; }
        .stButton>button:hover { background-color: #228B22; transform: scale(1.05); }
        .stSuccess { font-size: 18px; font-weight: bold; color: #1E90FF; }
        </style>
        """,
        unsafe_allow_html=True
    )

def load_pipelines():
    """Loads ML pipelines."""

    with open("../models/pipelines.pkl", "rb") as f:
        return pickle.load(f)

def get_user_input(model_option):
    """Handles user input based on selected prediction type."""

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with col2:
        height = st.number_input("Height (cm)", min_value=100, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=200)

    if model_option == "Calories to Burn":
        duration = st.number_input("Workout Duration (mins)", min_value=1, max_value=300)

        return {"Age": age, "Gender": gender, "Height": height, "Weight": weight, "Duration": duration}, "Calories"
    
    elif model_option == "Workout Duration":
        calories = st.number_input("Calories Burned", min_value=10, max_value=2000)

        return {"Age": age, "Gender": gender, "Height": height, "Weight": weight, "Calories": calories}, "Duration"

def validate_inputs(model_option, user_input):
    """Ensures user inputs are within dataset limits."""

    if model_option == "Calories to Burn" and user_input.get("Duration", 0) > 30:

        return "⚠️ Workout duration must be **30 minutes or less** for accurate predictions."
    
    elif model_option == "Workout Duration" and user_input.get("Calories", 0) > 300:

        return "⚠️ Calories burned must be **300 or less** for accurate predictions."
    
    return None

def make_prediction(model, input_data, target_feature):
    """Runs prediction and formats output."""

    prediction = model.predict(input_data)
    if target_feature == "Duration":
        minutes, seconds = divmod(int(prediction[0] * 60), 60)

        return f"✅ Estimated Workout Duration: **{minutes} mins {seconds} sec**"
    
    return f"✅ Estimated {target_feature}: **{prediction[0]:.2f}**"