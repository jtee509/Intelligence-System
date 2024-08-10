import numpy as np
import matplotlib.pyplot as plt

# Define the decision boundary equation
def decision_boundary(x1):
    return 2 - 0.5 * x1

# Generate x values
x1 = np.linspace(-10, 10, 100)

# Calculate corresponding x2 values
x2 = decision_boundary(x1)

# Plot the decision boundary
plt.plot(x1, x2, label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary of Logistic Regression Model')
plt.legend()
plt.show()