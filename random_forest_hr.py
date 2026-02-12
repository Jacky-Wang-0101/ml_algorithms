import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Generate Fake HR Dataset (Realistic Mode) ---
# "Pattern + Noise": There is a clear rule, but some people break it.

np.random.seed(42)
n_employees = 1000

data = {
    'satisfaction_level': np.random.uniform(0, 1, n_employees),
    'last_evaluation': np.random.uniform(0, 1, n_employees),
    'number_project': np.random.randint(2, 7, n_employees),
    'average_montly_hours': np.random.randint(90, 310, n_employees),
    'time_spend_company': np.random.randint(2, 10, n_employees),
    'work_accident': np.random.randint(0, 2, n_employees),
    'department': np.random.choice(['sales', 'technical', 'support', 'IT', 'hr'], n_employees),
    'salary': np.random.choice(['low', 'medium', 'high'], n_employees)
}
df = pd.DataFrame(data)

# --- Define a Clear Pattern (Signal) ---
# People who are overworked AND unhappy are LIKELY to leave.
# (Satisfaction < 0.5 AND Hours > 240)
conditions = (df['satisfaction_level'] < 0.5) & (df['average_montly_hours'] > 240)

# Apply the rule (This creates a clean separation)
df['left'] = np.where(conditions, 1, 0)

# --- Add Noise (The "Realistic" Part) ---
# In real life, patterns aren't perfect.
# We randomly flip 15% of the data to confuse the model.
noise_indices = np.random.choice(n_employees, size=int(n_employees * 0.15), replace=False)

# Flip the labels for these noisy people (0->1, 1->0)
df.loc[noise_indices, 'left'] = 1 - df.loc[noise_indices, 'left']

print("--- Data Preview ---")
print(f"Turnover Rate: {df['left'].mean():.2%}")

# --- 2. Preprocessing (Crucial Step!) ---
# Machines understand numbers, not words like "sales" or "low".
# We use "One-Hot Encoding" to turn categories into numbers.
# e.g., salary_low=1, salary_high=0
df_encoded = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)

# Separate Features (X) and Target (y)
X = df_encoded.drop('left', axis=1) # All columns except 'left'
y = df['left'] # The target variable

# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# --- 3. Build Random Forest ---
print("\n--- Training Random Forest ---")
# n_estimators=100 -> We are planting 100 trees!
model =RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 10}, random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 5. Feature Importance (The "Why") ---
# This is the most valuable part for business!
importances = model.feature_importances_
feature_names = X.columns

# Sort them
indices = np.argsort(importances)[::-1]

print("\n--- Key Drivers of Employee Churn ---")
for i in range(len(indices)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
