import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import difflib

class HardCodedClassifier:
    """
    The Hard Coded Classifier
    """

    def fit(self, data_train, targets_train):
        """
        Fit the data
        """
        return self

    def predict(self, data_test):
        targets_predicted = []

        for x in data_test:
            targets_predicted.append(0)

        return targets_predicted

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print "Iris Data"
print(iris.data)
print ""

# above and beyond to save and load data to and from a CSV file
np.savetxt("iris-data.csv", iris.data, delimiter=",")
iris_csv_data = genfromtxt('iris-data.csv', delimiter=',')
print "Iris CSV data"
print(iris_csv_data)
print ""

# Show the target values (in numeric format) of each instance
print "Iris Target"
print(iris.target)
print ""

# Show the actual target names that correspond to each number
print "Iris Target Names"
print(iris.target_names)
print ""

# above and beyond for cross validation
clf = svm.SVC(kernel='linear', C=1)
data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7)

# obtain scores from cross validation
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

classifier = GaussianNB()

model = classifier.fit(data_train, targets_train)

targets_predicted = model.predict(data_test)

print targets_predicted
print ""
print targets_test

sm=difflib.SequenceMatcher(None, targets_predicted, targets_test)
print "The two are " + str(sm.ratio()) + " percent similar"

classifier = HardCodedClassifier()
hard_model = classifier.fit(data_train, targets_train)
targets_predicted = hard_model.predict(data_test)

sm=difflib.SequenceMatcher(None, targets_predicted, targets_test)
print "The two are " + str(sm.ratio()) + " percent similar"

print "Scores:"
print scores