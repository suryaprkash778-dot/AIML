import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("housing.csv")

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to numpy arrays and cast to float
X_train_np = X_train.values.astype(float)
X_test_np = X_test.values.astype(float)
y_train_np = y_train.values.astype(float)

# Add bias term (intercept)
X_train_np = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
X_test_np = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])

# Least squares solution (use pseudoinverse for safety)
w = np.linalg.pinv(X_train_np) @ y_train_np

# Predictions
y_pred = X_test_np @ w

# Evaluation metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Show coefficients
print("Intercept:", w[0])
print("Coefficients:", w[1:])
