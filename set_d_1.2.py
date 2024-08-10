# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Create pandas dataframe
df = pd.DataFrame({
    "Year": [1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    "Male": [63.1, 63.5, 63.3, 63.8, 61.6, 62.6, 62.8, 63.2, 63.6, 64.3, 64.7, 65.3, 65.6, 65.8, 66.4, 66.9, 67.1, 67.1, 67.2, 67.7, 68.2, 68.5, 68.7, 68.8, 68.9, 69.2, 69.4, 69.6, 69.6, 69.5, 69.5, 69.7, 69.5, 69.7, 70, 70.6, 70.7, 70.8, 71.1, 71.4, 71.6, 71.6, 71.6, 71.7, 71.9, 72.1, 72.2, 72.4, 72.5, 72.5]
})

# Train linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[["Year"]], df["Male"])

# Predict life expectancy for 2025
year = 2025
predicted_life_expectancy = model.predict([[year]])[0]

# Plot the graph
plt.figure(figsize=(10, 6))
plt.scatter(df["Year"], df["Male"], color='blue', label='Actual Life Expectancy')
plt.plot(df["Year"], model.predict(df[["Year"]]), color='red', label='Predicted Life Expectancy')
plt.xlabel("Year")
plt.ylabel("Life Expectancy (Years)")
plt.title("Life Expectancy at Birth in Malaysia (Male)")
plt.axhline(y=predicted_life_expectancy, color='green', linestyle='--', label=f'Predicted for 2025: {predicted_life_expectancy:.2f}')
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted life expectancy for male in year 2025: {predicted_life_expectancy:.2f} years")