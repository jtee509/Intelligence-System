import csv
import polars as pl
import pandas as pd # data processing
import matplotlib.pyplot as plt #plotting library
from datetime import datetime #for time taken to converge

# Initialize an empty list to store the data
data_to_export = []

# Load the dataset
dataset = pl.read_excel('SET_D.xlsx')

#feature scaling (DISCOVERY : Feature scaling is necessary for learning rate = 0.01)
mean_Year = dataset['Year'].mean()
max_Year = dataset['Year'].max()
min_Year = dataset['Year'].min()# # Calculate the range
range_Year = max_Year - min_Year
dataset = dataset.with_columns((pl.col('Year') - mean_Year) / (range_Year))

Year = dataset['Year'] #X
Male = dataset['Male'] #Y


#QUESTION b

#Compute the Cost Function 𝐽(𝜃) by initialize the thetas all zeros.  
theta0 = 0
theta1 = 0

#start code
#so our theta will only be 2, theta0 and theta1, because we only have 1 x value
#hypothesis function
def h(x, c, m):
    return m * x + c

no_example = len(Year) #number of training examples

#cost function
def compute_cost(t0, t1, training_example):
    sum_mse = 0
    for i in range(training_example):
        x = Year[i]
        y = Male[i]
        sum_mse += float((h(x, t0, t1) - y)**2)
    cost = float(sum_mse / float(2 * training_example))
    return cost

#Implement Gradient Descent from scratch in Python. (Using learning rate, α = 0.01).
alpha1 = 0.01
alpha2 = 0.1
alpha3 = 0.001

def gradient_descent(t0, t1, training_ex, learning_rate):
    derived_cost_func_t0 = 0
    derived_cost_func_t1 = 0

    for i in range(training_ex):
        x = Year[i]
        y = Male[i]

        #derivative of cost function for theta 0 (gradient)
        derived_cost_func_t0 += (h(x, t0, t1) - y)

        #derivative of cost function for theta 1 (gradient)
        derived_cost_func_t1 += ((h(x, t0, t1) - y) * x)

    t0 = t0 - (float(learning_rate/(training_ex)) * derived_cost_func_t0)
    t1 = t1 - (float(learning_rate/(training_ex)) * derived_cost_func_t1)

    return t0, t1

epochs = 50 #fixed number of iterations
# if epoch_counter == epochs:
    #     endtime = datetime.now()
    #     break
epoch_counter = 1 #initialize the epoch counter to 1, which is to record the number of iterations
old_cost_function = compute_cost(theta0, theta1, no_example) + 1
endtime = datetime.now()
starttime = datetime.now()

while True: 
    cost_function = compute_cost(theta0, theta1, no_example)
    print(f"Cost function: ", cost_function)
    print(f"Theta 0: {theta0}, Theta 1: {theta1}")
    print(f"Epoch: {epoch_counter}")
    #store the cost function, theta0 and theta1 in a list and export as file
    data_to_export.append([cost_function, theta0, theta1, epoch_counter])
    if (old_cost_function - cost_function) <= 0.001:
        endtime = datetime.now()
        break
    # if epoch_counter == epochs:
    #     endtime = datetime.now()
    #     break
    theta0, theta1 = gradient_descent(theta0, theta1, no_example, alpha3)
    epoch_counter += 1  
    old_cost_function = cost_function
    
print(f"Time taken to converge: {endtime - starttime}")

#export the data to a csv file
with open('output_0.001_withoutScaled.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Cost Function", "Theta0", "Theta1", "Iteration"])  # Write the header
    writer.writerows(data_to_export)  # Write the data

#close file
f.close()

'''
#Learning rate = 0.001
# Theta 0: 1188.3561162706883, Theta 1: 394.01675572852025
# Epoch: 146142 (No of iterations)

# [Done] exited with code=1 in 28988.664 seconds
'''

'''
learning rate = 0.001, without feature scaling
Theta 0: 1166.6236985329836, Theta 1: 394.69857782939937
Epoch: 100118
Time taken to converge: 0:10:33.937653
'''

'''
learning rate = 0.001, with feature scaling
Cost function:  70388968.63952981
Theta 0: 13270.422265140349, Theta 1: 14603.08032248091
Epoch: 222176
Time taken to converge: 0:45:48.821383
'''

'''
learning rate = 0.01, with feature scaling
Cost function:  70388951.90701218
Theta 0: 13270.422265141167, Theta 1: 14628.507146594815
Epoch: 26497
Time taken to converge: 0:01:57.611912
'''
#HAVE YOU GOT THE tetha0 and thetha1 ?
#NOW GO TO plotting001.py (0.001) or plotting01.py (0.01) or plotting1.py (0.1) to plot the graph.