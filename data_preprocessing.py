from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import dataset
dataset = pd.read_csv('data/Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print('init X:', X)
print('init y: ', y)

# take care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print('after missing data X:', X)

# encode categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print('after encoded X:', X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print('y labeled: ', y)

# splitting dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print('after scaling X_train', X_train)
print('after scaling X_test', X_test)
