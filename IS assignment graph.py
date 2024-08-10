import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from the Excel file
df = pd.read_excel('SET_D.xlsx')

# Extract the columns for plotting
x = df['Year']
y = df['Male']

# Reshape x to make it a 2D array
x = x.values.reshape(-1, 1)

# Fit a linear regression model
model = LinearRegression()
model.fit(x, y)

# Make predictions using the model
predictions = model.predict(x)

# Plot the scatter plot
plt.scatter(x, y, label='Data Points')

# Plot the trend line
plt.plot(x, predictions, color='red', label='Trend Line')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Life Expectancy Age')
plt.title('Graph of Life Expectancy at birth by Sex (Male)')

# Add legend
plt.legend()

# Show plot
plt.show()