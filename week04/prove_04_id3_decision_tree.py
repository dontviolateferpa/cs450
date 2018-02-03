"""Implement ID3 algorithm to make a decision tree"""

import argparse
import pandas as pd

class MyDecisionTreeClassifier():
    """my class for the Decision Tree"""
    def __init__(self):
        pass
    
    def fit(self, train_data, train_targets):
        """fit the data"""
        return DTCModel(train_data, train_targets)

class DTCModel:
    _train_data = None
    _train_targets = None

    def __init__(self, train_data, train_targets):
        """put the data in the model"""
        self._train_data = train_data
        self._train_targets = train_targets

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

    cols = None
    cols = ["Credit Score", "Income", "Collateral", "Should Loan"]

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
    df_target = df['Should Loan']
    df.drop(columns=['Should Loan'], inplace=True)

    return df, df_target

def prep_data(args):
    """prepare the data from one of the datasets"""
    df = None
    df_target = None
    if args.csv_file == "id3_class.csv":
        df, df_target = prep_data_class_example(load_csv_file_class_example(args))
    else:
        raise ValueError("the script is not ready for this filename")

    return df, df_target

def main():
    """everything happens here"""

    args = receive_args().parse_args()
    df, df_target = prep_data(args)

    print df


main()