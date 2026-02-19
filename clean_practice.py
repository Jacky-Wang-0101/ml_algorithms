# ==========================================
# Data Cleaning Practice
# Goal: Fix the "dirty" health dataset
# ==========================================

import pandas as pd
import numpy as np

# 1. Load the dirty data
df = pd.read_csv("dirty_health_data.csv")

print("--- Before Cleaning (Head) ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())

# ==========================================
# Task 1: Clean Age (Outliers & Missing Values)
# ==========================================
# Step A: Remove unrealistic ages (e.g., > 120) by turning them into NaN
# Logic: Someone aged 250 is definitely a typo.
df.loc[df['Age'] > 120, 'Age'] = np.nan

# Step B: Calculate the mean age (ignoring NaNs automatically)
mean_age = df['Age'].mean()

# Step C: Fill missing values with the mean
df['Age'].fillna(mean_age, inplace=True)

# ==========================================
# Task 2: Standardize Gender (Inconsistent Labels)
# ==========================================
# Create a mapping dictionary to unify all variations
gender_map = {
    'Male': 'Male',
    'M': 'Male',
    'm': 'Male',
    'Man': 'Male',
    'Female': 'Female',
    'F': 'Female',
    'woman': 'Female'
}

# Apply the mapping
df['Gender'] = df['Gender'].map(gender_map)

# ==========================================
# Task 3: Fix Weight (String 'kg' & Unit Conversion)
# ==========================================
# Step A: Remove 'kg' string. We convert to str first to be safe.
df['Weight'] = df['Weight'].astype(str).str.lower().str.replace('kg', '')

# Step B: Convert column to numeric (float)
df['Weight'] = pd.to_numeric(df['Weight'])

# Step C: Convert lbs to kg
# Logic: If weight > 150, we assume it was recorded in lbs (since 150kg is rare)
# Formula: kg = lbs / 2.20462
mask_lbs = df['Weight'] > 150
df.loc[mask_lbs, 'Weight'] = df.loc[mask_lbs, 'Weight'] / 2.20462

# ==========================================
# Task 4: Fix Blood Pressure (Negative Values)
# ==========================================
# Step A: Identify negative values and turn them into NaN
df.loc[df['Blood_Pressure'] < 0, 'Blood_Pressure'] = np.nan

# Step B: Fill with mean (calculated from valid positive values)
bp_mean = df['Blood_Pressure'].mean()
df['Blood_Pressure'].fillna(bp_mean, inplace=True)

# ==========================================
# Task 5: Fix Glucose (Hidden Missing Values)
# ==========================================
# Step A: Replace -999 (sentinel value) with NaN
df['Glucose'] = df['Glucose'].replace(-999, np.nan)

# Step B: Fill with Median (Median is better for skewed biological data)
glucose_median = df['Glucose'].median()
df['Glucose'].fillna(glucose_median, inplace=True)

# ==========================================
# Final Check
# ==========================================
print("\n--- After Cleaning (Head) ---")
print(df.head())

print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# Save the clean version
df.to_csv("clean_health_data.csv", index=False)
print("\n✅ Data Cleaned and Saved as 'clean_health_data.csv'")