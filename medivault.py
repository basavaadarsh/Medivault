import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

try:
    # Load medical data
    medical_data = pd.read_csv('medical_data.csv')

    # Preprocess data
    X = medical_data.drop('DiseaseLabel', axis=1)  # Corrected column name
    y = medical_data['DiseaseLabel']  # Corrected column name

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Save the trained model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

except FileNotFoundError:
    print("Error: CSV file not found. Make sure the file exists in the correct directory.")
except Exception as e:
    print("An error occurred:", e)
