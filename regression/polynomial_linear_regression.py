import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import dataset
dataset = pd.read_csv('../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # matrix of featues
y = dataset.iloc[:, 2].values # linear dependent vector

# Fitting Linear Regression to the dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Linear Regression equation (y = b0 + b1*x1 + b2*x1^2 + ... + bn*x1^n)
polynomial_regressor = PolynomialFeatures(degree = 4)
X_poly = polynomial_regressor.fit_transform(X)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Salary According To Position')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regressor_2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Salary According To Position')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

linear_r_prediction = int(linear_regressor.predict([[6.5]]))
polynomial_r_prediction = int(linear_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]])))
print('Salary prediction with Linear Regression for level 6.5: ', linear_r_prediction)
print('Salary prediction with Polynomial Regression for level 6.5: ', polynomial_r_prediction)