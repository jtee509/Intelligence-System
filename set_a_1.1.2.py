import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



data = pd.read_csv('SET_A.csv')

mean_bmi = data['bmi'].mean()
mean_charges = data['charges'].mean()

numerator = sum((data['bmi'] - mean_bmi) * (data['charges'] - mean_charges))
denominator = sum((data['charges'] - mean_bmi) **  2)
slope = numerator / denominator
intercept = mean_charges - slope * mean_bmi

# Step  4: Predict the medical charge for a BMI of  27
bmi_27 =  27
predicted_charge = slope * bmi_27 + intercept
print(f"Predicted medical charge for BMI  27: {predicted_charge}")

# Step  5: Plot the original data points and the regression line
plt.scatter(data.bmi, data.charges, color='#00008B')
plt.plot([0, data['bmi'].max()], [slope * data['bmi'].min() + intercept, slope * data['bmi'].max() + intercept], color='red')
plt.xlabel('BMI')
plt.ylabel('Medical Charges')
plt.title('Linear Regression of Medical Charges vs BMI')
plt.show()