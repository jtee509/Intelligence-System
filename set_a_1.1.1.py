import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('SET_A.csv')

bmi = data['bmi']
charges = data['charges']

def loss_function(m, c, points):
    total_error = 0 
    for i in range (len(points)):
        x = points.iloc[i].bmi
        y = points.iloc[i].charges
        total_error += (y - (m * x + c)) ** 2
    total_error/ float(len(points))

def gradient_descent(m_now, c_now, points, L):
    m_gradient = 0
    c_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].bmi
        y = points.iloc[i].charges

        m_gradient += -(2/n) * x * (y - (m_now * x +c_now))
        c_gradient += -(2/n) * (y - (m_now * x +c_now))
    
    m = m_now - m_gradient * L        
    c = c_now - c_gradient * L  

    return m, c

m = 0
c = 0
L = 0.0001

epochs = 200

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch:{i}")
    m,b = gradient_descent(m,c,data,L)


X_predict = np.array([[27]])  # Put the BMI of which you want to predict charges here
y_predict = m.predict(X_predict)


print(m,b)

plt.scatter(bmi, charges, color='#000435', label='Original data')
plt.plot(list(range(data['bmi'])),[m * x + c for x in range(data['bmi'])], color="red")
plt.scatter(X_predict, y_predict, color='#FFA500', label='Prediction')
plt.xlabel('BMI')
plt.ylabel('Medical Charges')
plt.title('Linear Regression Model')
plt.show()