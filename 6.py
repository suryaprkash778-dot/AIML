import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('emails.csv')

# We'll isolate the numeric columns, as these techniques apply to them.
# We also want to exclude the identifier ('Email No.') and target ('Prediction') if they exist.
cols_to_exclude = ['Email No.', 'Prediction']
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in cols_to_exclude]

print(f"Original dataset shape: {df.shape}")

# ==========================================
# 1. HANDLING MISSING VALUES
# ==========================================
print("\n--- Missing Value Handling ---")
initial_missing = df.isnull().sum().sum()
print(f"Total missing values before processing: {initial_missing}")

# Technique: Median Imputation
# We fill any missing (NaN) numeric values with the median of their respective columns.
# (Filling with 0 or the mean are also valid alternatives here).
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

final_missing = df.isnull().sum().sum()
print(f"Total missing values after median imputation: {final_missing}")


# ==========================================
# 2. HANDLING OUTLIERS
# ==========================================
print("\n--- Outlier Handling ---")
# Technique: Capping with the Interquartile Range (IQR) Method
# Instead of deleting rows with outliers (which could delete too much data), 
# we "cap" the extreme values to an upper and lower boundary.

def cap_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Clip values so they don't exceed the bounds
    return series.clip(lower=lower_bound, upper=upper_bound)

print("Capping extreme outliers using the IQR method...")

# Apply the IQR capping function to all the numeric feature columns
for col in numeric_cols:
    df[col] = cap_outliers_iqr(df[col])

print("Outlier handling complete.")

# ==========================================
# FINAL OUTPUT
# ==========================================
print("\nPreprocessing for Experiment 6 is complete!")
print("Here is a peek at the cleaned data:")
print(df.head())
