from sklearn import metrics # importing evaluation matrices for regression from sklearn

#checking predictions made by model
predictions = lm.predict(X_test)
print(predictions)
sns.histplot((y_test-predictions))


print(metrics.mean_absolute_error(y_test,predictions)) #mean absolute error
print(metrics.mean_squared_error(y_test,predictions)) # mean squared error
print(metrics.root_mean_squared_error(y_test,predictions)) # root mean squared error

plt.show()
