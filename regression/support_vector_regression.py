import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# import dataset
dataset = pd.read_csv('../data/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting Support Vector Regression equation (y = b0 + b1*x1 + b2*x1^2 + ... + bn*x1^n)
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Visualising the Support Vector Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary According To Position using Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print('Salary prediction with Support Vector Regression for level 6.5: ', y_pred)
