from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_excel('SET_D.xlsx')

# Convert the 'Year' column to datetime
data['Year'] = pd.to_datetime(data['Year'], errors='coerce')

# Filter the data for males
male_data = data[['Year', 'Male']]

# Extract the year part and convert it to float
male_data['Year'] = male_data['Year'].dt.year.astype(float)

male_data['Male'] = male_data['Male'].astype(float)

# Reshape the 'Year' column to a  2D array with one column
X = male_data['Year'].values.reshape(-1,  1)

# Reshape the 'Male' column to a  1D array
y = male_data['Male'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict the life expectancy in  2025
X_predict = np.array([[2025.0]])
y_predict = model.predict(X_predict)

print(f'Predicted life expectancy in  2025 for males: {y_predict[0]}')

# Plot the data points
plt.scatter(male_data['Year'].astype(int), male_data['Male'].astype(float), marker='x', s=50, alpha=0.5)

# Calculate the regression line
y_fit = model.predict(X)

# Plot the regression line using the same X values as the model
plt.plot(male_data['Year'].astype(int), y_fit.astype(float), color="red")
plt.scatter(X_predict, y_predict, color='#FFA500', label='Prediction')
plt.legend()
plt.show()