import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce_Customers.csv')

#print(customers.head())
#print(customers.info())
#print(customers.describe())

#sns.jointplot(y="Yearly Amount Spent",x="Time on Website",data=customers, kind="scatter")
#sns.jointplot(y="Yearly Amount Spent",x="Time on App",data=customers, kind="scatter")
#sns.jointplot(y="Length of Membership",x="Time on App",data=customers, kind="hex")
#sns.pairplot(customers)
#plt.show()


y=customers['Yearly Amount Spent']
x=customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

lst = ['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

lm= LinearRegression()
lm.fit(X_train,y_train)

print("intercept: ",lm.intercept_)
print("coefficient: ",lm.coef_)

predicted = lm.predict(X_test)
#plt.scatter(y_test,predicted)
#plt.show()

print(metrics.mean_absolute_error(y_test,predicted))
print(metrics.mean_squared_error(y_test,predicted))
print(metrics.root_mean_squared_error(y_test,predicted))












