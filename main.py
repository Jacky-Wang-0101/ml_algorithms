from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
import os

# --- 1. Initialize App & Load Model ---
app = FastAPI()

# Load the trained model
# Ensure 'model_hr.pkl' is in the same directory, or this will fail.
model_path = 'model_hr.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
else:
    print("❌ Error: model_hr.pkl not found. Please run random_forest_hr.py first.")
    model = None

# --- 2. Define Input Data Schema ---
# This acts as a "form" that users must fill out.
# Pydantic validates that the data types are correct.
class EmployeeData(BaseModel):
    satisfaction_level: float  # 0.0 to 1.0
    last_evaluation: float  # 0.0 to 1.0
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    work_accident: int  # 0 or 1
    department: str  # 'sales', 'technical', 'support', 'IT', 'hr', etc.
    salary: str  # 'low', 'medium', 'high'

# --- 3. Define Prediction Endpoint (POST Request) ---
@app.post("/predict")
def predict_churn(employee: EmployeeData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # A. Create a DataFrame with all training features initialized to 0.
    # We must match the exact column names and order from training (X_train).
    # Based on pd.get_dummies(drop_first=True), the columns likely look like this:
    # Note: 'IT' might be dropped if it was the first category alphabetically, 
    # or 'marketing' etc. Adjust if your actual model has different columns.
    # For this exercise, we assume these standard dummy columns from the dataset.
    feature_columns = [
        'satisfaction_level', 
        'last_evaluation', 
        'number_project',
        'average_montly_hours', 
        'time_spend_company', 
        'work_accident',
        # Department dummy columns (one-hot encoded)
        'department_hr', 
        'department_sales', 
        'department_support', 
        'department_technical', 
        # Salary dummy columns
        'salary_low', 
        'salary_medium'
    ]

    # Create a single-row DataFrame filled with zeros
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    # B. Fill in numerical data
    input_data['satisfaction_level'] = employee.satisfaction_level
    input_data['last_evaluation'] = employee.last_evaluation
    input_data['number_project'] = employee.number_project
    input_data['average_montly_hours'] = employee.average_montly_hours
    input_data['time_spend_company'] = employee.time_spend_company
    input_data['work_accident'] = employee.work_accident

    # C. Handle Categorical Data (Manual One-Hot Encoding)
    # 1. Department
    # If user sends 'sales', we set 'department_sales' to 1.
    dept_col = f"department_{employee.department}"
    if dept_col in input_data.columns:
        input_data[dept_col] = 1

    # 2. Salary
    # If user sends 'low', we set 'salary_low' to 1.
    salary_col = f"salary_{employee.salary}"
    if salary_col in input_data.columns:
        input_data[salary_col] = 1

    # D. Make Prediction
    try:
        # predict() returns an array like [1] or [0]
        prediction = model.predict(input_data)[0]

        # predict_proba() returns probabilities like [[0.2, 0.8]]
        # We want the probability of class 1 (leaving)
        probability = model.predict_proba(input_data)[0][1]

        result = {
            "prediction": "Left" if prediction == 1 else "Stayed",
            "probability": round(probability, 2),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- 4. Health Check Endpoint ---
@app.get("/health")
def health_check():
    return {"status": "Model is ready", "model_loaded": model is not None}