import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('SET_D.xlsx')
data['Year'] = pd.to_datetime(data['Year'], errors='coerce')

# Normalize the data
data['Male'] = (data['Male'] - data['Male'].mean()) / data['Male'].std()

# Define the mean square error function
def mean_square_error(m, c, points):
    total_error =  0
    for i in range(len(points)):
        x = points.iloc[i]['Year'].year
        y = points.iloc[i]['Male']
        total_error += (y - (m * x + c)) **  2
    return total_error / float(len(points))

# Define the gradient descent function with gradient clipping
def gradient_descent(m_now, c_now, points, L, clip_value=1.0):
    m_gradient =   0
    c_gradient =   0
    n = len(points)
    for i in range(n):
        x = points.iloc[i]['Year'].year
        y = points.iloc[i]['Male']
        m_gradient += -(2/n) * x * (y - (m_now * x + c_now))
        c_gradient += -(2/n) * (y - (m_now * x + c_now))
    m_gradient = max(min(m_gradient, clip_value), -clip_value)  # Clip the gradient
    c_gradient = max(min(c_gradient, clip_value), -clip_value)  # Clip the gradient
    m = m_now - L * m_gradient        
    c = c_now - L * c_gradient   
    return m, c

# Initialize parameters
m =   0
c =   0
L =   0.01  # Reduced learning rate

# Perform gradient descent
epochs =   200
for i in range(epochs):
    if i %   50 ==   0:
        print(f"Epoch:{i}")
    m, c = gradient_descent(m, c, data, L)


# Predict life expectancy at birth for the year  2025
X_predict =  2025
y_predict = m * X_predict + c

print(f"Predicted life expectancy at birth for year  2025: {y_predict}")

data['Year'] = data['Year'].astype(int)
# Plot the data and the regression line
plt.scatter(data['Year'], data['Male'], color='#000435', label='Original data')
plt.plot(data['Year'], [m * x + c for x in data['Year']], color="red")
plt.scatter(X_predict, y_predict, color='#FFA500', label='Prediction')
plt.xlabel('Year')
plt.ylabel('Life Expectancy at Birth')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
