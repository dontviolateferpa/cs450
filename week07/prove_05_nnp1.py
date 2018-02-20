"""
Notice that at this point, we are NOT implementing any sort of training or learning
in our algorithm. The point of this part of the assignment is to make sure you have
the basics of a network of nodes in place that can handle a real instance from a
dataset.
"""

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS_IRIS = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"]
COLS_PIMA = ["Times Pregnant", "Plasma Glucose", "Diastolic Blood Pressure",
             "Triceps Skin Fold Thickness", "2-Hour Serum Insulin", "BMI",
             "Diabetes Pedigree Function", "Age", "Class Variable"]

class Network:
    """the Network contains nodes"""

    def __init__(self, sizes):
        """initialize the class"""
        # a "size" in the list of sizes specifies the number of neurons in each layer
        # of the network
        # the next three lines of code are from:
        #    https://bigsnarf.wordpress.com/2016/07/16/neural-network-from-scratch-in-python/
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.bias = -1
    
    def _add(self, input_vector):
        """sum the inputs"""
        keys = input_vector.keys()
        val_vector = []
        for key in keys:
            val_vector.append(input_vector[key])
        
        # got the following thre lines of code from
        #   https://bigsnarf.wordpress.com/2016/07/16/neural-network-from-scratch-in-python/
        for w in zip (self.weights[0]):
            a = sigmoid(np.dot(w, val_vector) + self.bias)
        return a
    
def sigmoid(v):
    """sigmoid function"""
    return 1/(1 + np.exp(-v))

def sigmoid_prime(v):
    """sigmoid prime function"""
    return sigmoid(v)*(1-sigmoid(v))

def receive_args():
    """receive arguments from the user pass to the script"""
    parser = argparse.ArgumentParser(description='Pass arguments to the script')
    parser.add_argument('--csv_file',
                        dest='csv_file',
                        action='store',
                        choices=["iris.csv", "pima.csv"],
                        required=True)
    parser.add_argument('--sizes',
                        dest='sizes',
                        action='store',
                        required=True,
                        nargs='+',
                        type=int)

    return parser

def check_args(args):
    """make sure the arguments passed by the user are valid"""
    if len(args.sizes) > 3:
        raise ValueError("too many layers for network--must be equal to or less than 3")
    for size in args.sizes:
        if size < 1:
            raise ValueError("incorrect number of nodes for a layer in the network--" +
                             "value must be greater than or equal to 1")

def load_csv_file_pima(args):
    """open csv file for pima indian dataset"""
    cols = COLS_PIMA
    df = pd.io.parsers.read_csv(
        args.csv_file,
        header=None,
        usecols=list(range(len(cols))),
        names=cols
    )
    return df

def prep_data_pima(df):
    """pepare the data for the pima indian dataset"""
    # this next line of code is from
    #   https://stackoverflow.com/questions/12525722/normalize-data-in-pandas
    df_target = df["Class Variable"]
    df.drop(columns=["Class Variable"], inplace=True)
    cols = df.columns
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df, df_target

def load_csv_file_iris(args):
    """open csv file for iris dataset"""
    cols = COLS_IRIS
    df = pd.io.parsers.read_csv(
        args.csv_file,
        header=None,
        usecols=list(range(len(cols))),
        names=cols
    )
    return df

def prep_data_iris(df):
    """prepare the data for the iris dataset"""
    # this next line of code is from
    #   https://stackoverflow.com/questions/12525722/normalize-data-in-pandas
    # df_norm = (df - df.mean()) / (df.max() - df.min())
    df_target = df["Class"]
    df.drop(columns=["Class"], inplace=True)
    cols = df.columns
    # for col in df.columns:
    #     df[col] = df.applymap(lambda x: (x - df[col].mean()) / (df[col].max() - df.min()))
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df, df_target

def prep_data(args):
    """prepare the data from one of the datasets"""
    df_data = None
    df_target = None
    if args.csv_file == "iris.csv":
        df_data, df_target = prep_data_iris(load_csv_file_iris(args))
    elif args.csv_file == "pima.csv":
        df_data, df_target = prep_data_pima(load_csv_file_pima(args))
    else:
        raise ValueError("the script is not ready for this filename")

    train_data, test_data, train_target, test_target = train_test_split(df_data, df_target, test_size=0.3)
    return train_data, test_data, train_target, test_target

def main():
    """where the magic happens"""
    args = receive_args().parse_args()
    check_args(args)
    train_data, test_data, train_target, test_target = prep_data(args)
    n = Network(args.sizes)
    for index, row in train_data.iterrows():
        # print len(row)
        # print row
        val = n._add(row)
        if val < 0.5:
            print 0
        else:
            print 1

main()