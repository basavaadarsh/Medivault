# Medivault
## Description 
The project described involves the development of a medical prediction system, named "MediVault", which utilizes machine learning models to predict whether a patient has a particular disease based on their medical data.

## Objective 
The primary objective of the project is to create a system that can predict whether a patient has a specific disease based on various medical parameters such as age, blood pressure, cholesterol level, and weight.

## Date Collection 
Synthetic medical data is generated for demonstration purposes. This data includes features like age, blood pressure, cholesterol level, weight, and a binary label indicating disease status (0 for no disease, 1 for disease).

## Model Training 
We Used Two machine learning algorithms, namely Logistic Regression and Decision Tree, are trained using the generated medical data. The models are trained to predict disease status based on the provided features.

## Web Application
A web-based interface is developed using the Streamlit library, allowing users to input their medical data and obtain predictions regarding disease status. The application provides users with the flexibility to choose between Logistic Regression and Decision Tree models for prediction.

## Model Deployement
The trained models are saved as pickle files (model.pkl and dt_model.pkl) along with the scaler (scaler.pkl). These files are loaded into the Streamlit application to make predictions in real-time.

## User Interaction
Users can interact with the web application by entering their medical data (age, blood pressure, cholesterol, weight) and selecting the desired machine learning model for prediction. Upon submission, the application displays the predicted disease status based on the selected model.

# Steps to run the code 

## Install Requirements.txt

```bash
pip install requirements.txt
```

## How to Run

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```

To exit the Streamlit application and stop the server, you can press Ctrl + C in the terminal.

## Output image 


![image](https://github.com/basavaadarsh/Medivault/assets/125342337/79d56a8c-0c17-47ab-9a61-8404febd95e5)





