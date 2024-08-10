import csv
import polars as pl #reading the dataset
import matplotlib.pyplot as plt #plotting library
from datetime import datetime #timestamp on comletion time

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

#for ploting graph
Year1 = dataset['Year'] #X
Male1 = dataset['Male'] #Y


#for data training
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

def feature_scalling(alpha,theta0, theta1):
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
        theta0, theta1 = gradient_descent(theta0, theta1, no_example, alpha)
        epoch_counter += 1   
        old_cost_function = cost_function
    
    print(f"Time taken to converge: {endtime - starttime}")

    #export the data to a csv file
    with open('output1_'+ str(alpha) +'_withoutScaled.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Cost Function", "Theta0", "Theta1", "Iteration"])  # Write the header
        writer.writerows(data_to_export)  # Write the data
    
    #close file
    f.close()
    return theta0, theta1

#Implement Gradient Descent from scratch in Python. (Using learning rate, α = 0.01).
alpha1 = 0.01
alpha2 = 0.1
alpha3 = 0.001

alphas = [0.01, 0.1, 0.001,0.0001]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

df = pl.read_excel('SET_D.xlsx')

# Iterate over alphas and subplots
for i, alpha in enumerate(alphas):
    # Assuming theta0 and theta1 are defined and feature_scalling is a function you've defined
    theta0, theta1 = feature_scalling(alpha, theta0, theta1)

    print('theta0 : ' + str(theta0) + '\ntheta1 : ' + str(theta1))

    line = theta0 + theta1 * Year

    # Determine the subplot to use based on the iteration index
    row = i // 2
    col = i % 2
    
    main_title = ('Life Expectancy \n Learning Rate of : ' + str(alpha))

    axs[row, col].set_title(label=main_title)

    # Plot the data and the best fit line on the current subplot
    axs[row, col].scatter(df['Year'], df['Male'], label='Original Data')
    axs[row, col].set_xlabel("Year")
    axs[row, col].set_ylabel("Male")
    axs[row, col].plot(df['Year'], line, color='red', label='Gradient Descent')

    x0 = Year.min() # Starting point of the line
    y0 = theta0 + theta1 * x0 # Calculate the corresponding y value

    # Draw the infinite line
    axs[row, col].axline((df['Year'].min(), line), slope=theta1, color='red', linestyle='--', label='Gradient Descent')

    # Predict charge if Year = 2025
    Year_test = ((2025 - mean_Year) / (range_Year))
    predicted_male = theta0 + theta1 * Year_test
    print(predicted_male)
    axs[row, col].scatter(2025, predicted_male, label = ('Prediction for 2025 : \n' + str(round(predicted_male,8))) )
    axs[row, col].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

# Show the plots
plt.show()