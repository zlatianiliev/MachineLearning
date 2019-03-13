import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# import dataset
dataset = pd.read_csv('../data/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)

# Fitting Decision Tree Regression
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary According To Position using Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

y_pred=regressor.predict(np.array([[6.5]]))
print('Salary prediction with Decision Tree for level 6.5: ', y_pred)
