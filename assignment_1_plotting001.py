import polars as pl #data reading for csv files
import matplotlib.pyplot as plt #plotting library

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

#plot based on the thetas you get in assignment_1.py
#without feature scaling
theta0 = 68.18167744249934
theta1 = 6.853818646659526


line = theta0 + theta1 * Year


# Plot the data and the best fit line
plt.scatter(Year, Male)
plt.xlabel("Year")
plt.ylabel("Male",)
plt.plot(Year, line, color='red', label='Predicted Life Expectancy')

#predict charge if Year = 27
Year_test = ((2025 - mean_Year) / (range_Year))
predicted_male = theta0 + theta1 * Year_test
print(predicted_male)

plt.scatter(Year_test, predicted_male)
plt.show()