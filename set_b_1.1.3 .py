import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('SET_B.csv')

height = data['Height(Inches)']
charges = data['Weight(Pounds)']

def mean_square_error(m, c, points):
    total_error =  0  
    for i in range(len(points)):
        x = points.iloc[i].height
        y = points.iloc[i].charges
        total_error += (y - (m * x + c)) **  2
    return total_error / float(len(points))

def gradient_descent(m_now, c_now, points, L):
    m_gradient =  0
    c_gradient =  0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].height
        y = points.iloc[i].charges

        m_gradient += -(2/n) * x * (y - (m_now * x + c_now))
        c_gradient += -(2/n) * (y - (m_now * x + c_now))
    
    m = m_now - m_gradient * L        
    c = c_now - c_gradient * L   

    return m, c

m =  0
c =  0
L =  0.0001

epochs =  200

for i in range(epochs):
    if i %  50 ==  0:
        print(f"Epoch:{i}")
    m, c = gradient_descent(m, c, data, L)

# Predict the medical charge for a height of  27
X_predict =  27
y_predict = m * X_predict + c

print(f"Predicted medical charge for height  27: {y_predict}")

plt.scatter(height, charges, color='#000435', label='Original data')
plt.plot(height, [m * x + c for x in height], color="red")
plt.scatter(X_predict, y_predict, color='#FFA500', label='Prediction')
plt.xlabel('height')
plt.ylabel('Medical Charges')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
