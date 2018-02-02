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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

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

    def predict(self, dt, is_np_array):
        """Make a prediction"""
        targets_predict = []
        data_test = dt
        data_train = self._data_train
        # we want to compute the nearest neighbors of data_test
        if is_np_array == False:
            for point_x in data_test:
                nns = []
                y_count = 0
                for point_y in data_train:
                    # compute euc distance
                    dist = np.linalg.norm(point_x - point_y)
                    nns.append([dist, self._targets_train[y_count]])
                    y_count = y_count + 1
                nns = sorted(nns)
                nn = self._compute_nn(nns)
                targets_predict.append(nn)
            # this will have to be the same length as targets_test
        else:
            # the reason for doing this is because the prepration from sklearn
            # turns the data into an sklearn data type
            for x, point_x in np.ndenumerate(data_test):
                point_y = []
                nns = []
                y_count = 0
                for tup_y, y in np.ndenumerate(data_train):
                    point_y.append(y)
                    if tup_y[0] == 5:
                        # compute euc distance
                        dist = np.linalg.norm(point_x - point_y)
                        nns.append([dist, self._targets_train.tolist()[y_count]])
                        y_count = y_count + 1
                        point_y = []
                nns = sorted(nns)
                nn = self._compute_nn(nns)
                targets_predict.append(nn)
        # this will have to be the same length as targets_test
        return np.array(targets_predict)

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

def mpg_data_to_array(data):
    """convert mpg's data_train to an array without tuples"""
    data_converted = []
    point = []
    for tup_y, y in np.ndenumerate(data):
        point.append(y)
        if tup_y[1] == 5:
            data_converted.append(point)
            point = []

    return data_converted

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
    parser.add_argument('--target_col', dest='target_col', action='store')
    parser.add_argument('--na_values', dest='na_values', action='store', nargs='*', type=str)
    parser.add_argument('--col_names', dest='col_names', action='store', nargs='*', type=str)
    parser.add_argument('--use_cols', dest='use_cols', action='store', nargs='*', default=[], type=int)

    return parser

def load_data_set_from_csv(file_name, args):
    """load the dataset from a csv"""
    column_names = None
    NA_VALUE = None
    is_whitespace = False

    CAR_COL_NAMES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'condition']
    PIMA_COL_NAMES = ['times pregnant', 'plasma glucose concentration', 'diastolic blood pressure', 'triceps skin fold thickness', '2H serum insulin', 'bmi', 'dpf', 'age', 'class var']
    MPG_COL_NAMES = ['mpg','cyl','disp','hp','weight','acc','year','origin','name']
    if file_name == "car.csv":
        column_names = CAR_COL_NAMES
    elif file_name == "pima.csv":
        column_names = PIMA_COL_NAMES
        na_value = float('nan')
    elif file_name == "mpg.csv":
        column_names = MPG_COL_NAMES
        is_whitespace = True
    else:
        raise ValueError("incorrect file name loaded to read csv file")

    df = pd.io.parsers.read_csv(
        file_name,
        header=None,
        usecols=list(range(len(column_names))),
        names=column_names,
        na_values=None,
        delim_whitespace=is_whitespace
    )

    return df

def prep_data_car(ds, args):
    """prepare the cars dataset"""
    ds.dropna(inplace=True)
    col_names = args.col_names
    drop_cols = args.drop_cols
    target_col = args.target_col
    # get classes we will predict
    ds_target = ds['condition'].astype('category').cat.codes
    # drop classes we will predict from data    
    ds.drop(columns=['condition'], inplace=True)
    ds['buying'] = ds['buying'].astype('category').cat.codes
    ds['maint'] = ds['maint'].astype('category').cat.codes
    ds['doors'] = ds['doors'].astype('category').cat.codes
    ds['persons'] = ds['persons'].astype('category').cat.codes
    ds['lug_boot'] = ds['lug_boot'].astype('category').cat.codes
    ds['safety'] = ds['safety'].astype('category').cat.codes
    return ds, ds_target

def prep_data_pima(ds, args):
    """prep the pima indians dataset"""
    ds[['plasma glucose concentration', 'diastolic blood pressure', 'triceps skin fold thickness', '2H serum insulin', 'bmi']] = ds[['plasma glucose concentration', 'diastolic blood pressure', 'triceps skin fold thickness', '2H serum insulin', 'bmi']].replace(0, np.NaN)
    print(ds.isnull().sum())
    ds.dropna(inplace=True)
    ds_target = ds['class var']
    ds.drop(columns=['class var'], inplace=True)
    return ds, ds_target

def prep_data_mpg(ds, args):
    """prep the mpg dataset"""
    lab_enc = LabelEncoder()
    ds[['cyl','disp','hp','weight','acc','year','origin']] = ds[['cyl','disp','hp','weight','acc','year','origin']].replace('?', np.NaN)
    ds.dropna(inplace=True)
    ds_target = ds['mpg'].astype('int64')
    ds.drop(columns=['mpg', 'name'], inplace=True)
    ds = pd.get_dummies(ds, columns=['origin'])
    ds['cyl'] = ds['cyl'].astype('float64')
    ds['year'] = ds['year'].astype('float64')
    ds['disp'] = ds['disp'].astype('float64')
    ds['hp'] = ds['hp'].astype('float64')
    ds['weight'] = ds['weight'].astype('float64')
    ds['acc'] = ds['acc'].astype('float64')

    # this caused me a lot of headaches, but it's necessary
    # this returns an sklearn datatype, not a pandas dataframe
    std_scale = preprocessing.StandardScaler().fit(ds[['cyl', 'disp', 'hp', 'weight', 'acc', 'year']])
    ds_std = std_scale.transform(ds[['cyl','disp', 'hp', 'weight', 'acc', 'year']])

    return ds_std, ds_target

def choose_data_set(args):
    """choose the dataset based on args passed"""
    df_data = None
    df_target = None
    if args.csv_file == None:
        raise ValueError("no file name passed")
    elif args.csv_file == "car.csv":        
        df_data, df_target = prep_data_car(load_data_set_from_csv(args.csv_file, args), args)
    elif args.csv_file == "pima.csv":
        df_data, df_target = prep_data_pima(load_data_set_from_csv(args.csv_file, args), args)
    elif args.csv_file == "mpg.csv":
        df_data, df_target = prep_data_mpg(load_data_set_from_csv(args.csv_file, args), args)
    else:
        raise ValueError("name of file must be either \'car.csv\', \'pima.csv\' or \'mpg.csv\'")
    
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, df_data, df_target)
    print "scores from cross validation"
    print scores

    return model_selection.train_test_split(df_data, df_target, test_size=0.3, random_state=3)


def test_classifier(classifier, args, method, data_train, data_test, targets_train, targets_test):
    """test a model"""
    model = None
    targets_predicted = None
    clf = svm.SVC(kernel='linear', C=1)
    scores = None

    # since we are working with two different data types, we have to branch the logic (ugly, I know)
    if args.csv_file != "mpg.csv":
        model = classifier.fit(data_train.as_matrix(columns=None), targets_train.as_matrix(columns=None))
        targets_predicted = model.predict(data_test.as_matrix(columns=None), False)
        display_similarity(targets_predicted, targets_test.as_matrix(columns=None), method)

        classifier = KNeighborsClassifier(n_neighbors=3)
        model = classifier.fit(data_train.as_matrix(columns=None), targets_train.as_matrix(columns=None))
        targets_predicted = model.predict(data_test.as_matrix(columns=None))
        display_similarity(targets_predicted, targets_test.as_matrix(columns=None), "KNearest")

    elif args.csv_file == "mpg.csv":
        model = classifier.fit(data_train, targets_train)
        targets_predicted = model.predict(data_test, True)
        display_similarity(targets_predicted, targets_test.as_matrix(columns=None), method)

        classifier = KNeighborsClassifier(n_neighbors=3)
        model = classifier.fit(data_train, targets_train)
        targets_predicted = model.predict(data_test)
        display_similarity(targets_predicted, targets_test.as_matrix(columns=None), "KNearest")

def display_similarity(predictions, targets_test, method):
    """display the similarity between two arrays"""
    sim_score = 0
    count = 0
    for x in predictions:
        if x == targets_test[count]:
            sim_score = sim_score + 1
    print "The two are " + str(float(sim_score) / float(len(predictions))) + " percent similar (" + method + ")"

def main():
    """main"""
    # receive arguments from the user
    args = receive_args().parse_args()
    # receive the data needed to classify
    dtr, dte, ttr, tte = choose_data_set(args)
    test_classifier(MyKNearestClassifier(3), args, "MyKNearest", dtr, dte, ttr, tte)

main()
