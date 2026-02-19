import pandas as pd
import numpy as np
import random

# Set seed for reproducibility (so we get the same "mess" every time)
np.random.seed(42)
random.seed(42)

n_samples = 500

data = {
    'Patient_ID': [f'P{i:03d}' for i in range(n_samples)],
    
    # 1. Missing Value Hell: Randomly lose 10% of Age data
    'Age': [np.random.randint(18, 90) if random.random() > 0.1 else np.nan for _ in range(n_samples)],
    
    # 2. String Chaos: Inconsistent gender labels
    'Gender': np.random.choice(['Male', 'Female', 'M', 'F', 'm', 'woman', 'Man'], n_samples),
    
    # 3. Unit Mess: Weights mixed with numbers, strings, and wrong units (lbs)
    'Weight': [],
    
    # 4. Logic Errors: Blood pressure with negative or impossible values
    'Blood_Pressure': [int(np.random.normal(120, 20)) for _ in range(n_samples)],
    
    # 5. Hidden Missing Values: Using -999 to represent missing data
    'Glucose': [int(np.random.normal(100, 30)) if random.random() > 0.05 else -999 for _ in range(n_samples)]
}

# Generate messy Weight data
for _ in range(n_samples):
    w = np.random.randint(50, 120)
    r = random.random()
    if r < 0.7:
        data['Weight'].append(w)        # Normal numeric (kg)
    elif r < 0.9:
        data['Weight'].append(f"{w}kg") # String with unit (e.g., "70kg")
    else:
        data['Weight'].append(w * 2.2)  # Wrong unit error (converted to lbs, so value is huge)

# Create extreme outliers manually
data['Age'][0] = 250  # Impossible age (Typo?)
data['Blood_Pressure'][1] = -10  # Negative blood pressure (Physically impossible)

df = pd.DataFrame(data)

# Save to CSV
filename = "dirty_health_data.csv"
df.to_csv(filename, index=False)

print(f"✅ Dirty data generated successfully: {filename}")
print("Your Mission: Clean this dataset using Python!")