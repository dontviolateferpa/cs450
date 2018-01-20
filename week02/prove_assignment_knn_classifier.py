import argparse
import difflib
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class HardCodedClassifier:
    """The Hard Coded Classifier"""
    def fit(self, data_train, targets_train):
        """Fit the data"""
        return self

    def predict(self, data_test):
        targets_predicted = []

        for x in data_test:
            targets_predicted.append(0)

        return targets_predicted

class MyKNearestClassifier:
    """My implementation of the k-nearest neighbors classifier."""
    _k = None
    def __init__(self, k):
        self._k = k

    def fit(self, data_train, targets_train):
        """Fit the data"""
        return KModel(self._k, data_train, targets_train)

class KModel:
    """A model with data and a predict method"""
    _k = None
    _data_train = None
    _targets_train = None

    def __init__(self, k, data_train, targets_train):
        """Put data in the model"""
        self._k = k
        self._data_train = data_train
        self._targets_train = targets_train

    def predict(self, data_test):
        """Make a prediction"""
        targets_predict = []

        # we want to compute the nearest neighbors of data_test
        for point_x in data_test:
            nns = []
            y_count = 0
            for point_y in self._data_train:
                # compute euc distance
                dist = np.linalg.norm(point_x - point_y)
                nns.append([dist, self._targets_train[y_count]])
                y_count = y_count + 1
            nns = sorted(nns)
            nn = self._compute_nn(nns)
            targets_predict.append(nn)

        # this will have to be the same length as targets_test
        return targets_predict

    def _compute_nn(self, nns):
        """Compute the nearest neighbor"""
        top_k = nns[:self._k]
        k_classes = get_col(top_k, 1)
        nn = None
        frequency = 0
        for x in k_classes:
            if k_classes.count(x) >= frequency:
                frequency = k_classes.count(x)
                nn = x
        return nn

def get_col(arr, col):
    """
    Got this function from Stack Overflow
    https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array
    """
    return map(lambda x : x[col], arr)

def receive_args():
    """pass arguments to the script"""
    parser = argparse.ArgumentParser(description='Pass arguments to the script')
    parser.add_argument('--csv_file', dest='csv_file', action='store', default=None)
    parser.add_argument('--save_file', dest='save_file', action='store', default=None)

    return parser

def load_data_set():
    """load dataset"""
    iris = datasets.load_iris()

    return iris

def load_data_set_from_csv(file_name):
    """load the dataset from a csv"""
    # above and beyond to save (commented out) and load data to and from a CSV file
    iris_csv_data = genfromtxt(file_name, delimiter=',')

def save_csv(file_name):
    """save a csv file"""
    np.savetxt(file_name, iris.data, delimiter=",")

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

def display_similarity(predictions, targets_test, method):
    """display the similarity between two arrays"""
    sm=difflib.SequenceMatcher(None, predictions, targets_test)
    print "The two are " + str(sm.ratio()) + " percent similar (" + method + ")"

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

    display_similarity(targets_predicted, targets_test, "gaussian")

    classifier = HardCodedClassifier()
    hard_model = classifier.fit(data_train, targets_train)
    targets_predicted = hard_model.predict(data_test)

    display_similarity(targets_predicted, targets_test, "hard coded")

    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)

    display_similarity(predictions, targets_test, "k-nearest neighbors")

    classifier = MyKNearestClassifier(3)
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)

    display_similarity(predictions, targets_test, "my k-nearest neighbors")
    
    print "Scores:"
    print scores

    if args.save_file != None:
        save_csv(args.save_file)

main()