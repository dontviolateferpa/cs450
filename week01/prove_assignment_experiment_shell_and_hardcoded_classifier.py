from sklearn import datasets
from sklearn.model_selection import train_test_split
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
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7)

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
