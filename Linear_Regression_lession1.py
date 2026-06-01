import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('USA_Housing.csv')

# Check columns
print(df.columns)

# Define features (exclude 'Price' and 'Address')
X = df[['Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population']]

# Define target
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Fit model
lm = LinearRegression()
lm.fit(X_train, y_train)


# Print intercept and coefficients
print("Intercept:", lm.intercept_)
print("Coefficients:", lm.coef_)

cdf=pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
print(cdf)








