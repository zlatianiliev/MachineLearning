import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# import dataset
dataset = pd.read_csv('../data/50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encode categorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# splitting dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Add a column of ones in order to make the Multiple Linear Regression equation
# (y = b0*1 + b1*x1 + b2*x2 + ... + bn*xn) have a const b0 because the statsmodels lib doesn't include it
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Building the optimal model using Backward Elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        # OLS = ordinary least squares, method for estimating the unknown parameters
        # in a linear regression model
        regressor_OLS = sm.OLS(y, x).fit()
        # find the biggest P values and compare to SL if P > SL -> remove the independant variable
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
print(X_Modeled)
