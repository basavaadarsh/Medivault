import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load(r'C:\Users\basav\OneDrive\Desktop\medivault\model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess input data and make predictions
def predict_disease(age, blood_pressure, cholesterol, weight):
    # Preprocess the input data (Scale after checking)
    input_data = pd.DataFrame({'Age': [age], 'BloodPressure': [blood_pressure], 'Cholesterol': [cholesterol], 'Weight': [weight]})
    input_data_scaled = scaler.transform(input_data) if scaler else input_data
    
    # Make predictions
    prediction = model.predict(input_data_scaled)
    
    return prediction[0]

# Streamlit app
def main():
    st.title('MediVault Prediction')

    # Input form for user to input medical data
    age = st.number_input('Age', min_value=0, max_value=150, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=300, step=1)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, step=1)
    weight = st.number_input('Weight', min_value=0, max_value=300, step=1)

    # Prediction button
    if st.button('Predict'):
        prediction = predict_disease(age, blood_pressure, cholesterol, weight)
        st.write(f'Prediction: {"You are having some disease please attend to hospital" if prediction == 1 else "Negative you are having good health and having no disease"}')

if __name__ == '__main__':
    main()
