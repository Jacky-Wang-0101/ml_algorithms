# ==========================================
# XGBoost Training Script (The Kaggle Killer)
# Goal: Beat Random Forest's performance on Noisy Data
# ==========================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# --- 1. Generate the SAME Realistic Data (Pattern + Noise) ---
# We use the same logic as yesterday to ensure fair comparison
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

# The Logic: Overworked (hours > 240) AND Unhappy (satisfaction < 0.5)
conditions = (df['satisfaction_level'] < 0.5) & (df['average_montly_hours'] > 240)
df['left'] = np.where(conditions, 1, 0)

# The Noise: Flip 15% of labels (simulating real-world chaos)
noise_indices = np.random.choice(n_employees, size=int(n_employees * 0.15), replace=False)
df.loc[noise_indices, 'left'] = 1 - df.loc[noise_indices, 'left']

print(f"Data Generated. Turnover Rate: {df['left'].mean():.2%}")

# --- 2. Preprocessing (One-Hot Encoding) ---
# XGBoost handles numbers well, but strictly needs all inputs to be numeric
df_encoded = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)

X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train XGBoost Model ---
# Calculate Scale Pos Weight for Imbalance Handling
# Formula: sum(negative_instances) / sum(positive_instances)
# This tells XGBoost: "Pay X times more attention to the minority class (leavers)"
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    n_estimators=100,      # Number of boosting rounds (trees)
    learning_rate=0.1,     # Step size shrinkage (lower is slower but more robust)
    max_depth=5,           # Depth of each tree (higher = more complex)
    scale_pos_weight=scale_pos_weight, # Crucial for handling imbalance!
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("--- Training XGBoost ---")
model.fit(X_train, y_train)

# --- 4. Evaluate ---
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- 5. Save the New Brain ---
filename = 'model_xgboost.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved as {filename}")