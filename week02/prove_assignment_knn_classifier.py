import argparse
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

def receive_args():
    """pass arguments to the script"""
    parser = argparse.ArgumentParser(description='Pass arguments to the script')
    parser.add_argument('--csv_file', dest='csv_file', action='store', default=None)

    return parser

def load_data_set():
    """load dataset"""
    iris = datasets.load_iris()

    return iris

def load_data_set_from_csv(file_name):
    """load the dataset from a csv"""
    # above and beyond to save (commented out) and load data to and from a CSV file
    # np.savetxt("iris-data.csv", iris.data, delimiter=",")
    iris_csv_data = genfromtxt(file_name, delimiter=',')

def choose_data_set(args):
    """choose the dataset based on args passed"""
    data_set = None
    if args.csv_file == None:
        data_set = load_data_set()
    else:
        data_set = load_data_set_from_csv(args.csv_file)

    return data_set

def print_data_set(data_set):
    """print the data"""

    # Show the data (the attributes of each instance)
    print "data"
    print (data_set.data)
    print ""

    # Show the target values (in numberic format) of each instance
    print "target"
    print (data_set.target)
    print ""

    # Show the actual target names that correspond to each number
    print "target names"
    print(data_set.target_names)
    print ""

def main():
    """main"""
    args = receive_args().parse_args()
    iris = choose_data_set(args)
    print_data_set(iris)

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

main()