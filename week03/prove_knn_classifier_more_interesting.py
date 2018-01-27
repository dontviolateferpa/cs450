import argparse
import difflib
import numpy as np
import pandas as pd
import re
from numpy import genfromtxt
from numpy.linalg import norm
from sklearn import datasets
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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
        for point_x in data_test.iterrows():
            nns = []
            y_count = 0
            # for point_y in self._data_train.iterrows():
            #     # compute euc distance
            #     print "point_y"
            #     print point_y
            #     dist = np.linalg.norm(point_x - point_y)
            #     nns.append([dist, self._targets_train[y_count]])
            #     y_count = y_count + 1
            # nns = sorted(nns)
            # nn = self._compute_nn(nns)
            # targets_predict.append(nn)
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
    parser.add_argument('--csv_file', dest='csv_file', action='store', required=True)
    parser.add_argument('--drop_cols', dest='drop_cols', action='store', nargs='*', type=int)
    parser.add_argument('--target_col', dest='target_col', action='store', required=True)
    parser.add_argument('--na_values', dest='na_values', action='store', nargs='*', type=str)
    parser.add_argument('--col_names', dest='col_names', action='store', nargs='*', type=str)
    parser.add_argument('--use_cols', dest='use_cols', action='store', nargs='*', default=[], type=int)

    return parser

def load_data_set_from_csv(file_name, args):
    """load the dataset from a csv"""
    df = pd.io.parsers.read_csv(
        file_name,
        header=None,
        usecols=list(range(len(args.col_names))),
        names=args.col_names,
        na_values=args.na_values
    )

    return df

def prep_data(ds, args):
    """prepare the dataset we receive"""
    ds.dropna(inplace=True)
    # get classes we will predict
    ds_target = ds.get(args.target_col).astype('category').cat.codes
    # drop classes we will predict from data    
    ds_data = ds.drop(columns=args.target_col)
    cols_dict_g = ds_data.columns.to_series().groupby(ds_data.dtypes).groups
    cols_dict = {k.name: v for k, v in cols_dict_g.items()}
    # get all cols of type 'object'
    cols = cols_dict['object']
    # make all object-type cols have dummy cols
    ds_data = pd.get_dummies(ds_data, columns=cols)
    return ds_data, ds_target

def choose_data_set(args):
    """choose the dataset based on args passed"""
    df_data = None
    df_target = None
    if args.csv_file == None:
        raise ValueError("no file name passed")
    else:        
        df_data, df_target = prep_data(load_data_set_from_csv(args.csv_file, args), args)
    return model_selection.train_test_split(df_data, df_target, test_size=0.3, random_state=3)


def test_classifier(classifier, data_train, data_test, targets_train, targets_test):
    """test a model"""
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)

    # display_similarity(targets_predicted, targets_test, "MyKNearestNeighbor")

    # print "data_train"
    # print data_train.tail()
    # print "data_test"
    # print data_test.tail()
    # print "targets_train"
    # print targets_train.head()
    # print "targets_test"
    # print targets_test.head()

def display_similarity(predictions, targets_test, method):
    """display the similarity between two arrays"""
    sm=difflib.SequenceMatcher(None, predictions, targets_test)
    print "The two are " + str(sm.ratio()) + " percent similar (" + method + ")"

def main():
    """main"""
    args = receive_args().parse_args()
    choose_data_set(args)
    dtr, dte, ttr, tte = choose_data_set(args)
    test_classifier(MyKNearestClassifier(3), dtr, dte, ttr, tte)

main()
