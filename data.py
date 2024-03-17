import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic medical data
num_samples = 1000

age = np.random.randint(20, 80, num_samples)
blood_pressure = np.random.randint(80, 180, num_samples)
cholesterol = np.random.randint(120, 300, num_samples)
weight = np.random.randint(40, 120, num_samples)

# Generating disease labels: 0 for no disease, 1 for disease
disease_label = np.random.randint(0, 2, num_samples)

# Create DataFrame
medical_data = pd.DataFrame({
    'Age': age,
    'BloodPressure': blood_pressure,
    'Cholesterol': cholesterol,
    'Weight': weight,
    'DiseaseLabel': disease_label
})

# Save the dataset to a CSV file
medical_data.to_csv('medical_data.csv', index=False)
