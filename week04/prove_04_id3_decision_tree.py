"""Implement ID3 algorithm to make a decision tree"""

import argparse
import difflib
import pandas as pd
import numpy as np
from anytree import Node, RenderTree, AsciiStyle, Walker, LevelOrderIter, Resolver
from anytree.exporter import DictExporter
from anytree import find
from collections import Counter
from sklearn.model_selection import train_test_split
import search

COLS_CLASS_EXAMPLE = ["Credit Score", "Income", "Collateral", "Should Loan"]
COLS_CLASS_EXAMPLE_TRAIN = ["Credit Score", "Income", "Collateral"]
COLS_IRIS = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"]
COLS_IRIS_TRAIN = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

class MyDecisionTreeClassifier:
    """my class for the Decision Tree"""
    def __init__(self):
        pass
    
    def fit(self, train_data, train_target):
        """fit the data"""
        return DTCModel(train_data, train_target)

class DTCModel:
    _train_data = None
    _train_target = None
    _tree = None
    _target_feature = None

    def __init__(self, train_data, train_target):
        """put the data in the model"""
        self._train_data = train_data
        self._train_target = train_target
        self._target_feature = train_target.to_frame().columns[0]
        classes = []
        self._tree = self._make_ID3_tree(self._train_data, self._train_target, None, "root")
        print RenderTree(self._tree, style=AsciiStyle())

    def _calc_entropy(self, numers, denom):
        """calc the entropy for this particular attribute's value"""
        total_entropy = 0
        for numer in numers:
            p = numer / float(denom)
            total_entropy -= (p * np.log2(p))

        return total_entropy

    def _calc_entropies_aux(self, array_dicts):
        """calculate the entropy of each attribute, helper function"""
        entropies = []
        for i_dict in array_dicts:
            numers = []
            weight = 0
            sum_weight = 0
            sum_entropy = 0
            for key in i_dict:
                # this will be needed to compute the average entropy between values of attributes
                weight = len(i_dict[key])
                for key, value in Counter(i_dict[key]).iteritems():
                    numers.append(value)
                # make a sum of total entropy
                sum_entropy += self._calc_entropy(numers, weight) * weight
                sum_weight += weight
                if float(sum(numers)) / weight > 1:
                    raise ValueError("sum of the numers exceeds size of denom: (%d/%d)" % (sum(numers), weight))
                numers = []
            entropy = sum_entropy / float(sum_weight)
            # make a list of our entropies
            entropies.append(entropy)

        return entropies

    def _calc_entropies(self, train_data, train_target):
        """calculate the entropy of each attribute"""
        buckets = []
        feats = []
        cidx = 0
        for feat in train_data.columns:
            # store the column index in along with the column name
            buckets.append({})
            feats.append(feat)
            for ridx, row in train_data[feat].iteritems():
                if row in buckets[cidx]:
                    buckets[cidx][row].append(train_target[ridx])
                else:
                    buckets[cidx][row] = []
                    buckets[cidx][row].append(train_target[ridx])
            cidx += 1
        # turn this into a dictionary of column names with their values as the entropies
        entropies_dict = {}
        entropies = self._calc_entropies_aux(buckets)
        e_count = 0
        for entropy in entropies:
            entropies_dict[feats[e_count]] = entropy
            e_count += 1
        return entropies_dict

    def _make_ID3_tree(self, train_data, train_target, node, value):
        """make the ID3 decision tree"""
        # if there are no features left to split on
        if len(train_data.columns) == 0:
            return Node(value, parent=node, target=train_target.mode().as_matrix()[0], feat=None)
        # if all rows in feature have the same target (entropy == 0)
        elif train_target[train_target == train_target.as_matrix()[0]].count() == len(train_target):
            return Node(value, parent=node, target=train_target.as_matrix()[0], feat=None)
        elif len(train_data.index) == 0:
            raise ValueError("we have not handled this base case yet")
        else:
            entropies = self._calc_entropies(train_data, train_target)
            # find the lowest value in the key-value pairs
            feat_max_info_gain = min(entropies, key=entropies.get)
            n = Node(value, parent=node, feat=feat_max_info_gain)
            # get all unique possible values of a feature
            feat_values = train_data[feat_max_info_gain].unique()
            fv_array = []
            potential_feat_values = self._train_data[feat_max_info_gain].unique()
            pfv_array = []
            # make up targets for datapoints the training set does not have
            for val in feat_values:
                fv_array.append(val)
            for val in potential_feat_values:
                pfv_array.append(val)
            missing_feat_values = list(set(pfv_array) - set(fv_array))
            whole_df_joined = self._train_data.join(self._train_target, lsuffix='_data', rsuffix='_target')
            for missing_value in missing_feat_values:
                # this is where we make up the target
                child = Node(missing_value, parent=n, target=whole_df_joined[whole_df_joined[feat_max_info_gain] == missing_value][train_target.to_frame().columns[0]].mode()[0])
            df_joined = train_data.join(train_target, lsuffix='_data', rsuffix='_target')
            for value in feat_values:
                df_joined_subset = df_joined[df_joined[feat_max_info_gain] == value]
                df_joined_target_subset = df_joined_subset[self._target_feature]
                df_joined_subset.drop(columns=[self._target_feature, feat_max_info_gain], inplace=True)
                self._make_ID3_tree(df_joined_subset, df_joined_target_subset, n, value)
        return n

    def _get_class(self, row):
        """get the class of a row in a dataframe"""
        node = self._tree
        r = Resolver('name')
        while not node.is_leaf:
            if node.feat != None:
                print "the " + node.feat + " is " + row[node.feat]
                n = r.get(node, row[node.feat])
                node = n
        print "\tso we predict " + node.target
        return node.target

    def predict(self, test_data):
        """predict the classes of the test data"""
        predicted_targets = []
        for idx, row in test_data.iterrows():
            predicted_targets.append(self._get_class(row))
        
        return predicted_targets

def receive_args():
    """pass arguments to the script"""
    parser = argparse.ArgumentParser(description='Pass arguments to the script')
    parser.add_argument('--csv_file',
                        dest='csv_file',
                        action='store',
                        choices=["id3_class.csv", "iris.csv"],
                        required=True)

    return parser

def load_csv_file_class_example(args):
    """open csv file"""
    cols = COLS_CLASS_EXAMPLE
    df = pd.io.parsers.read_csv(
        args.csv_file,
        header=None,
        usecols=list(range(len(cols))),
        names=cols,
        delim_whitespace=True
    )
    return df

def prep_data_class_example(df):
    """prepare the data for the class example set"""
    df["Credit Score"] = df["Credit Score"]
    df["Income"] = df["Income"]
    df["Collateral"] = df["Collateral"]
    df["Should Loan"] = df["Should Loan"]
    df_target = df['Should Loan']
    df.drop(columns=['Should Loan'], inplace=True)

    return df, df_target

def load_csv_file_iris(args):
    """open csv file"""
    cols = COLS_IRIS
    df = pd.io.parsers.read_csv(
        args.csv_file,
        header=None,
        usecols=list(range(len(cols))),
        names=cols
    )
    return df

def prep_data_iris(df):
    """prepare the data for the iris data set"""
    df["Sepal Length"] = pd.cut(df["Sepal Length"], 3, labels=["Short", "Med", "Long"])
    df["Sepal Width"] = pd.cut(df["Sepal Width"], 3, labels=["Thin", "Med", "Thick"])
    df["Petal Length"] = pd.cut(df["Petal Length"], 3, labels=["Short", "Med", "Long"])
    df["Petal Width"] = pd.cut(df["Petal Width"], 3, labels=["Thin", "Med", "Thick"])
    df.to_csv('iris-df.csv')
    df_target = df["Class"]
    df.drop(columns=["Class"], inplace=True)   

    return df, df_target

def prep_data(args):
    """prepare the data from one of the datasets"""
    df_data = None
    df_target = None
    if args.csv_file == "id3_class.csv":
        df_data, df_target = prep_data_class_example(load_csv_file_class_example(args))
    elif args.csv_file == "iris.csv":
        df_data, df_target = prep_data_iris(load_csv_file_iris(args))
    else:
        raise ValueError("the script is not ready for this filename")

    train_data, test_data, train_target, test_target = train_test_split(df_data, df_target, test_size=0.3)
    return train_data, test_data, train_target, test_target

def display_similarity(predictions, targets_test, method):
    """display the similarity between two arrays"""
    if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
        predictions = predictions.as_matrix()
    if isinstance(targets_test, pd.DataFrame) or isinstance(targets_test, pd.Series):
        targets_test = targets_test.as_matrix()
    sm=difflib.SequenceMatcher(None, predictions, targets_test)
    print "The two are " + str(sm.ratio()) + " percent similar (" + method + ")"

def test_node(parent, name):
    return Node(name, parent=parent)

def main():
    """everything happens here"""
    args = receive_args().parse_args()
    train_data, test_data, train_target, test_target = prep_data(args)
    model = MyDecisionTreeClassifier().fit(train_data, train_target)
    predicted_targets = model.predict(test_data)
    display_similarity(predicted_targets, test_target, "Decision Tree")
    
    # n = Node("n")
    # a = test_node(n, "a")
    # b = test_node(n, "b")
    # c = test_node(b, "c")

    # print RenderTree(n)

    # what the class example should look like
    # print "\n--> CLASS EXAMPLE BY HAND <--"
    # z = Node("", parent=None, feat="Income")
    # child_a1 = Node("High", parent=z)
    # child_a2 = Node("Low", parent=z)
    # child_a1.feat="Credit Score"
    # child_b1 = Node("Good", parent=child_a1)
    # child_b2 = Node("Average", parent=child_a1)
    # child_b3 = Node("Low", parent=child_a1)
    # child_b3.feat="Collateral"
    # child_c1 = Node("Good", parent=child_b3)
    # child_c2 = Node("Poor", parent=child_b3)
    # child_c1.target="Yes"
    # child_c2.target="No"
    # child_b1.target="Yes"
    # child_b2.target="Yes"
    # print RenderTree(z)

main()