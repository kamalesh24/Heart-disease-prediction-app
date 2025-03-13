import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the XGBoost model
model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    xgb_model = pickle.load(f)

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = xgb_model.predict(input_df)
    return prediction

# Streamlit user interface
st.set_page_config(page_title='Heart Disease Prediction App', page_icon=':heart:', layout='wide')
st.title('Heart Disease Prediction App')
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# User input fields
st.sidebar.header('User Input Parameters')
def user_input_features():
    age = st.sidebar.slider('Age', min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: ['Asymptomatic', 'Atypical Angina', 'Non-Anginal', 'Typical Angina'][x])
    trestbps = st.sidebar.slider('Resting Blood Pressure', min_value=90, max_value=200, value=120)
    chol = st.sidebar.slider('Cholesterol', min_value=131, max_value=409, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], format_func=lambda x: ['Left Ventricular Hypertrophy', 'Normal', 'ST-T'][x])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=71, max_value=199, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: ['Downsloping', 'Flat', 'Upsloping'][x])
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0)
    thal = st.sidebar.selectbox('Thalassemia', options=[0, 1, 2], format_func=lambda x: ['Fixed', 'Normal', 'Reversible'][x])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_data)

# Prediction button
if st.button('Predict'):
    prediction = predict(input_data)
    prediction_labels = {
        0: "No Heart Disease",
        1: "Arrhythmia",
        2: "Cardiomyopathy",
        3: "Congenital Heart Disease",
        4: "Coronary Artery Disease",
        5: "Heart Failure",
        6: "Valvular Heart Disease"
    }
    prediction_text = prediction_labels.get(prediction[0], "Unknown")
    st.subheader('Prediction')
    st.write(f"Prediction: **{prediction_text}**")
    
    from gradio_client import Client
    advice_prompt = f"""A patient has been evaluated based on the following medical parameters:

                        Age: {age} years
                        Sex: {'Male' if sex == 1 else 'Female'}
                        Chest Pain Type: {['Asymptomatic', 'Atypical Angina', 'Non-Anginal', 'Typical Angina'][cp]}
                        Resting Blood Pressure: {trestbps} mmHg
                        Cholesterol Level: {chol} mg/dL
                        Fasting Blood Sugar > 120 mg/dL: {'Yes' if fbs == 1 else 'No'}
                        Resting Electrocardiographic Results: {['Left Ventricular Hypertrophy', 'Normal', 'ST-T'][restecg]}
                        Maximum Heart Rate Achieved: {thalach} bpm
                        Exercise Induced Angina: {'Yes' if exang == 1 else 'No'}
                        ST Depression Induced by Exercise: {oldpeak}
                        Slope of the Peak Exercise ST Segment: {['Downsloping', 'Flat', 'Upsloping'][slope]}
                        Number of Major Vessels Colored by Fluoroscopy: {ca}
                        Thalassemia Type: {['Fixed', 'Normal', 'Reversible'][thal]}
                        The patient has been diagnosed with {heart_disease}.

                        Please provide a concise 3-line recommendation on how the patient can manage or improve their condition, including lifestyle changes, dietary adjustments, and medical advice."""
                          
    client = Client("KingNish/Very-Fast-Chatbot")
    result = client.predict(
            Query=advice_prompt,
            api_name="/predict"
    )
    result = result.strip().replace("\n", "\n\n")  # Double newline = markdown list-friendly in Streamlit

    st.markdown("### 📋 Recommended Next Steps")
    st.markdown(f"""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #0073e6; border-radius: 8px; margin-top: 10px;">
                <span style="font-size: 16px;">{result}</span>
            </div>
        """, unsafe_allow_html=True)