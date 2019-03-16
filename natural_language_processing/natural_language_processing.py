import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('../data/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
nltk.download('stopwords')
ps = PorterStemmer()
N = len(dataset)
corpus = [] # collection of text of the same type
for i in range(0, N):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    # TODO: need to keep the negative words like 'not' and use n-grams
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# [TP][FP]
# [FN][TN]

cm = confusion_matrix(y_test, y_pred)

# Printing the Accuracy, Precision, Recall and F1 Score
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 Score = 2 * Precision * Recall / (Precision + Recall)

total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
correctly_classified = cm[0][0] + cm[1][1]
accuracy = correctly_classified / total
print('Accuracy: ', accuracy)

precision = cm[0][0] / (cm[0][0] + cm[0][1])
print('Precision: ', precision)

recall = cm[0][0] / (cm[0][0] + cm[1][0])
print('Recall: ', recall)

f1_score = 2 * precision * recall / (precision + recall)
print('F1 Score: ', f1_score)
