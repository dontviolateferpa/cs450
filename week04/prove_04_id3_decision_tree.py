"""Implement ID3 algorithm to make a decision tree"""

import argparse
import pandas as pd

def receive_args():
    """pass arguments to the script"""
    parser = argparse.ArgumentParser(description='Pass arguments to the script')
    parser.add_argument('--csv_file', dest='csv_file', action='store', choices=["id3_class.csv"] required=True)

    return parser

def load_csv_file_class_example(args):
    """open csv file"""

    cols = None
    cols = ["CreditScore", "Income", "Collateral", "ShouldLoan"]

    df = pd.io.parsers.read_csv(
        args.csv_file,
        header=None
    )
    pass

def prep_data_class_example(df):
    """prepare the data for the class example set"""
    pass

def prep_data(args):
    """prepare the data from one of the datasets"""

    if args.csv_file == "id3_class.csv":
        load_csv_file_class_example(args):
    else:
        raise ValueError("the script is not ready for this filename")

def main():
    """everything happens here"""

    args = receive_args().parse_args()


main()