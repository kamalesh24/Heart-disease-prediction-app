import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the XGBoost model efficiently
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

xgb_model = load_model()

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = xgb_model.predict(input_df)
    return prediction

# Streamlit UI Enhancements
st.set_page_config(page_title='Heart Disease Prediction', page_icon='❤️', layout='wide')

st.title('💖 Heart Disease Prediction & Advice')
st.markdown("**Get AI-powered insights and personalized recommendations for heart health!**")

# Sidebar Styling
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2947/2947762.png", width=120)
st.sidebar.header('🔍 User Input Parameters')

# Collecting User Inputs
def user_input_features():
    age = st.sidebar.slider('Age', 0, 120, 30)
    sex = st.sidebar.radio('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3], format_func=lambda x: ['Asymptomatic', 'Atypical Angina', 'Non-Anginal', 'Typical Angina'][x])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mmHg)', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dL)', 131, 409, 200)
    fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dL?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    restecg = st.sidebar.selectbox('Resting ECG Results', [0, 1, 2], format_func=lambda x: ['LV Hypertrophy', 'Normal', 'ST-T Wave'][x])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 199, 150)
    exang = st.sidebar.radio('Exercise-Induced Angina?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('ST Segment Slope', [0, 1, 2], format_func=lambda x: ['Downsloping', 'Flat', 'Upsloping'][x])
    ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia Type', [0, 1, 2], format_func=lambda x: ['Fixed', 'Normal', 'Reversible'][x])

    # Structuring Data
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    # Generating a structured prompt for recommendations
    advice_prompt = f"""A patient has been evaluated with these health metrics:
    
    - **Age:** {age} years
    - **Sex:** {'Male' if sex == 1 else 'Female'}
    - **Chest Pain Type:** {['Asymptomatic', 'Atypical Angina', 'Non-Anginal', 'Typical Angina'][cp]}
    - **Resting Blood Pressure:** {trestbps} mmHg
    - **Cholesterol Level:** {chol} mg/dL
    - **Fasting Blood Sugar > 120 mg/dL:** {'Yes' if fbs == 1 else 'No'}
    - **Resting ECG Results:** {['LV Hypertrophy', 'Normal', 'ST-T Wave'][restecg]}
    - **Max Heart Rate Achieved:** {thalach} bpm
    - **Exercise Induced Angina:** {'Yes' if exang == 1 else 'No'}
    - **ST Depression:** {oldpeak}
    - **ST Segment Slope:** {['Downsloping', 'Flat', 'Upsloping'][slope]}
    - **Major Vessels Colored by Fluoroscopy:** {ca}
    - **Thalassemia Type:** {['Fixed', 'Normal', 'Reversible'][thal]}
    
    Please provide expert medical advice on managing this condition. 

    """
    return input_data, advice_prompt

input_data, advice_prompt = user_input_features()

# Display User Inputs in an Expandable Section
with st.expander("📌 View Your Entered Details"):
    st.write(pd.DataFrame(input_data, index=[0]))

# Prediction Button
if st.button('🩺 Predict Heart Disease'):
    prediction = predict(input_data)
    diagnosis = {
        0: "No Heart Disease ✅",
        1: "Arrhythmia ⚠️",
        2: "Cardiomyopathy ⚠️",
        3: "Congenital Heart Disease ⚠️",
        4: "Coronary Artery Disease ❗",
        5: "Heart Failure ❗",
        6: "Valvular Heart Disease ❗"
    }
    result = diagnosis.get(prediction[0], "Unknown")

    # Show Prediction Result
    if prediction[0] == 0:
        st.success(f"**Diagnosis: {result}**\n\nYour heart is healthy! Keep up with a balanced diet and regular exercise. 💙")
    else:
        st.error(f"**Diagnosis: {result}**\n\nYou may need medical attention. Consult a cardiologist. 🏥")

    # AI-Powered Health Advice
    from gradio_client import Client
    client = Client("KingNish/Very-Fast-Chatbot")
    medical_advice = client.predict(Query=advice_prompt + f" Diagnosed with {result}. Provide a professional recommendation.", api_name="/predict")

    # Display Medical Advice
    st.subheader("🩺 AI-Generated Medical Advice")
    st.write(medical_advice)

# Footer with credits
st.markdown("---")
st.markdown("📝 Developed by **Kamalesh S** | 🔗 [GitHub](https://github.com/HariPrashand)")
