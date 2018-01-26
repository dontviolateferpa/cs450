import argparse
import difflib
import numpy as np
import pandas as pd
import re
from numpy import genfromtxt
from numpy.linalg import norm
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# regex for comma separated list of integers
# ((\d+)$(\d+,)*)

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
    parser.add_argument('--csv_file', dest='csv_file', action='store', default=None, required=True)
    parser.add_argument('--save_file', dest='save_file', action='store', default=None)
    parser.add_argument('--drop_cols', dest='drop_cols', action='store', default=None)
    parser.add_argument('--na_value', dest='na_value', action='store', default=" ?")

    return parser

def check_args(args):
    """make sure args are valid"""
    pass

def load_data_set_from_csv(file_name):
    """load the dataset from a csv"""
    df = pd.io.parsers.read_csv(
        file_name,
        header=None,
        na_values=[" ?"]
    )

    return df

def prep_data(ds):
    """prepare the dataset we receive"""
    ds.dropna(inplace=True)
    cols_dict_g = ds.columns.to_series().groupby(ds.dtypes).groups
    cols_dict = {k.name: v for k, v in cols_dict_g.items()}
    # get all cols of type 'object'
    cols = cols_dict['object']
    # make all object-type cols have dummy cols
    ds = pd.get_dummies(ds, columns=cols)
    return ds

def save_csv(file_name):
    """save a csv file"""
    np.savetxt(file_name, iris.data, delimiter=",")

def choose_data_set(args):
    """choose the dataset based on args passed"""
    data_set = None
    if args.csv_file == None:
        raise ValueError("no file name passed")
    else:
        data_set = prep_data(load_data_set_from_csv(args.csv_file))

    return data_set

def display_similarity(predictions, targets_test, method):
    """display the similarity between two arrays"""
    sm=difflib.SequenceMatcher(None, predictions, targets_test)
    print "The two are " + str(sm.ratio()) + " percent similar (" + method + ")"

def main():
    """main"""
    args = receive_args().parse_args()
    check_args(args)

    # print "Dataset:"
    df = choose_data_set(args)
    # print df

    if args.save_file != None:
        save_csv(args.save_file)

main()
