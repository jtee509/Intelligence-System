import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('SET_A.csv')

# Normalize the features
scaler = StandardScaler()
data[['bmi']] = scaler.fit_transform(data[['bmi']])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data['bmi'], data['charges'], test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train.values.reshape(-1,  1), y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val.values.reshape(-1,  1))

# Calculate performance metrics
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Print the performance metrics
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Predict the medical charge for BMI  27
bmi_27 = np.array([27]).reshape(-1,  1)
predicted_charge = model.predict(bmi_27)
print(f"Predicted medical charge for BMI  27: {predicted_charge[0]}")

# Plot the data points and the regression line
plt.scatter(X_val, y_val, color="blue")
plt.plot(X_val, model.predict(X_val.values.reshape(-1,  1)), color="red")
plt.plot(predicted_charge,marker="o", color="red")
plt.xlabel('BMI')
plt.ylabel('Medical Charges')
plt.title('Linear Regression Model')
plt.show()
