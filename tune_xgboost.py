# ==========================================
# Hyperparameter Tuning Script (GridSearchCV)
# Goal: Stop guessing parameters, let the computer find the best ones.
# ==========================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# --- 1. Generate Realistic Data (Same as yesterday) ---
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

# Logic + Noise
conditions = (df['satisfaction_level'] < 0.5) & (df['average_montly_hours'] > 240)
df['left'] = np.where(conditions, 1, 0)
noise_indices = np.random.choice(n_employees, size=int(n_employees * 0.15), replace=False)
df.loc[noise_indices, 'left'] = 1 - df.loc[noise_indices, 'left']

# Encoding
df_encoded = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Define the "Grid" (The Combinations to Try) ---
param_grid = {
    'n_estimators': [50, 100, 200],      # How many trees?
    'max_depth': [3, 5, 7],              # How deep is each tree?
    'learning_rate': [0.01, 0.1, 0.2],   # How fast does it learn?
    'scale_pos_weight': [3]              # Keep this fixed for imbalance
}

# Total combinations = 3 * 3 * 3 * 1 = 27 models to train!

# --- 3. Setup GridSearchCV ---
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

print("--- Starting Grid Search (This might take a minute...) ---")

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='recall',  # <--- WE OPTIMIZE FOR RECALL (Catching leavers is priority!)
    cv=3,              # Cross-Validation: Split data into 3 parts to verify
    verbose=1          # Print progress
)

# --- 4. Run the Search ---
grid_search.fit(X_train, y_train)

# --- 5. The Moment of Truth ---
print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)

print("\n📊 Evaluation with BEST parameters:")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))