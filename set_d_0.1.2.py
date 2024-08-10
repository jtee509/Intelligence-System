from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('SET_A.csv')

bmi = data['bmi']
charges = data['charges']

# Reshape bmi to a  2D array with one column
bmi = bmi.values.reshape(-1,  1)

model = LinearRegression()
model.fit(bmi, charges)

X_predict = np.array([[27]])  # Put the BMI of which you want to predict charges here
y_predict = model.predict(X_predict)

plt.scatter(bmi, charges, color='#000435', label='Original data')

# Optionally, plot the prediction for the given BMI
plt.scatter(X_predict, y_predict, color='#FFA500', label='Prediction')

# Plot the regression line
plt.plot(bmi, model.predict(bmi), color='red', label='Fitted line')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('Linear Regression')
plt.legend()
plt.show()