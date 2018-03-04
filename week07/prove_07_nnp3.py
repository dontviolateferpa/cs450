"""
Notice that at this point, we are NOT implementing any sort of training or learning
in our algorithm. The point of this part of the assignment is to make sure you have
the basics of a network of nodes in place that can handle a real instance from a
dataset.
"""

import argparse
import difflib
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS_IRIS = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"]
COLS_PIMA = ["Times Pregnant", "Plasma Glucose", "Diastolic Blood Pressure",
             "Triceps Skin Fold Thickness", "2-Hour Serum Insulin", "BMI",
             "Diabetes Pedigree Function", "Age", "Class Variable"]
COLS_EXPL = ["x1", "x2", "c"]

class MLP:
    """the Network contains nodes"""
    def __init__(self, sizes, classes, example=False):
        """initialize the class"""
        # a "size" in the list of sizes specifies the number of neurons in each layer
        # of the network
        # the next four lines of code are from:
        #    https://bigsnarf.wordpress.com/2016/07/16/neural-network-from-scratch-in-python/
        self.activations  = []
        self.learning_rate = 0.1
        if example == False:
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
            self.bias_weights = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            self.num_layers = 3
            self.sizes = [2, 2, 2]
            #                w_i1 = nodes i to node 1 in hidden layer1
            #                w_i1_ol = nodes i to node 1 in output layer
            #                w_i1_h1     w_i2_h1        w_i1_ol      w_i2_ol
            self.weights = [[[0.3, 0.4], [0.2, -0.1]], [[0.2, -0.1], [-0.2, 0.4]]]
            self.bias_weights = [[[0.2], [-0.1]], [[0.3], [0.1]]]
        self.classes = classes
        # there should only be one output for each hidden node coming from the bias, hence
        #   the dimension is (y, 1) [y being the number of nodes receiving input from the bias]

    def _feed_forward(self, input_vector):
        """sum the inputs"""
        # got the following thre lines of code from
        #   https://bigsnarf.wordpress.com/2016/07/16/neural-network-from-scratch-in-python/
        keys = input_vector.keys()
        a = []
        self.activations = []
        for key in keys:
            a.append(input_vector[key])
        self.activations.append(input_vector)
        # got the following thre lines of code from
        #   https://bigsnarf.wordpress.com/2016/07/16/neural-network-from-scratch-in-python/
        for b, w in zip (self.bias_weights, self.weights):
            a = np.dot(w,a)
            for index in range(len(a)):
                a[index] += (-1)*b[index][0]
                a[index] = sigmoid(a[index])
            self.activations.append(a)
        return a
    
    def _feed_backward(self, predict_tar, correct_tar, r_weights, r_bias_weights, r_activations):
        """feed backward"""
        pt, ct, rw, rw, ra = predict_tar, correct_tar, r_weights, r_bias_weights, r_activations
        print ">>>>>>> feed backwards <<<<<<<"
        r_errors = []
        for i in range(0,len(r_weights)):
            e_layer = []
            for j in range(len(r_activations[i])):
                e, a, o = None, r_activations[i][j], 0
                if j == np.where(self.classes==correct_tar)[0]:
                    o = 1 # expected target
                if i == 0: # output layer
                    e = a*(1-a)*(a-o)
                    e_layer.append(e)
                else: # hidden layer
                    temp_weights = []
                    for k in range(len(r_weights[i])):
                        temp_weights.append(r_weights[i-1][k][j])
                    e = a*(1-a)*np.dot(temp_weights, r_errors[i-1])
                    e_layer.append(e)
            r_errors.append(e_layer)
        for ir in range(len(self.weights))[::-1]: # iterate over layers
            for j in range(len(self.weights)): #iterate over nodes
                wij = self.weights[ir][j]
                ai = self.activations[ir][j]
                dj = r_errors[len(self.weights)-ir-1][j]
                print "\tdj"
                print dj
                print "\twij"
                print wij
                print "\tai"
                print ai
                # print wij - self.learning_rate * dj
                print self.bias_weights[ir][j]

            print ""
        return r_errors[::-1]

    def predict(self, test_data):
        """make predictions on the dataset"""
        predicted_class, predictions = None, []
        for index, row in test_data.iterrows():
            predicted_class = self._feed_forward(row)
            predictions.append(self.classes[np.argmax(predicted_class)])
        return predictions
    
    def fit(self, train_data, train_target):
        """fit the classifier"""
        predicted_class, predictions = None, []
        for tar, row in zip(train_target.to_frame().iterrows(), train_data.iterrows()):
            predicted_class = self._feed_forward(row[1])
            # POSSIBLE ISSUE WITH np.argmax
            predictions.append(self.classes[np.argmax(predicted_class)])
            predict_tar = self.classes[np.argmax(predicted_class)]
            correct_tar = tar[1].to_frame().as_matrix()[0][0]
            self._feed_backward(predict_tar, correct_tar, self.weights[::-1], self.bias_weights[::-1],
                                self.activations[::-1])
        return predictions

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
                        choices=["iris.csv", "pima.csv", "class_example.csv"],
                        required=True)
    parser.add_argument('--sizes',
                        dest='sizes',
                        action='store',
                        required=True,
                        nargs='+',
                        type=int)

    return parser

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
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df, df_target

def load_csv_file_example(args):
    """open csv file for iris dataset"""
    cols = COLS_EXPL
    df = pd.io.parsers.read_csv(
        args.csv_file,
        header=None,
        usecols=list(range(len(cols))),
        names=cols
    )
    return df

def prep_data_example(df):
    """prepare the data for the class example"""
    df_target = df["c"]
    df.drop(columns=["c"], inplace=True)    
    return df, df_target

def prep_data(args):
    """prepare the data from one of the datasets"""
    df_data = None
    df_target = None
    if args.csv_file == "iris.csv":
        df_data, df_target = prep_data_iris(load_csv_file_iris(args))
    elif args.csv_file == "pima.csv":
        df_data, df_target = prep_data_pima(load_csv_file_pima(args))
    elif args.csv_file == "class_example.csv":
        df_data, df_target = prep_data_example(load_csv_file_example(args))
        return df_data, None, df_target, None
    else:
        raise ValueError("the script is not ready for this filename")

    train_data, test_data, train_target, test_target = train_test_split(df_data, df_target, test_size=0.3)
    return train_data, test_data, train_target, test_target

def prep_sizes(data, target, hidden_nodes):
    """prepare an array of sizes for the nodes in the network"""
    hn_copy = hidden_nodes
    hn_copy.insert(0, data.shape[1])
    hn_copy.append(len(target.to_frame()[target.to_frame().columns[0]].unique()))
    return hn_copy

def display_similarity(predictions, targets_test, method):
    """display the similarity between two arrays"""
    if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
        predictions = predictions.as_matrix()
    if isinstance(targets_test, pd.DataFrame) or isinstance(targets_test, pd.Series):
        targets_test = targets_test.as_matrix()
    sm=difflib.SequenceMatcher(None, predictions, targets_test)
    print "The two are " + str(sm.ratio()) + " percent similar (" + method + ")"

def main():
    """where the magic happens"""
    args = receive_args().parse_args()
    train_data, test_data, train_target, test_target = prep_data(args)
    df_target = pd.concat([train_target, test_target])
    possible_classes = df_target.to_frame()[df_target.to_frame().columns[0]].unique()
    n_classifier = None
    if args.csv_file != "class_example.csv":
        sizes = prep_sizes(train_data, pd.concat([train_target, test_target]), args.sizes)
        n_classifier = MLP(sizes, possible_classes)
        # CHANGE THIS TO TEST DATA IN THE FUTURE
        predictions = n_classifier.fit(train_data, train_target)
        display_similarity(predictions, test_target, "Neural Network")        
    else:
        n_classifier = MLP(args.sizes, possible_classes, example=True)
        predictions = n_classifier.fit(train_data, train_target)
        display_similarity(predictions, train_target, "Neural Network")

main()