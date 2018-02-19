"""
Notice that at this point, we are NOT implementing any sort of training or learning
in our algorithm. The point of this part of the assignment is to make sure you have
the basics of a network of nodes in place that can handle a real instance from a
dataset.
"""

import argparse
import numpy as np

class Network:
    """the Network contains nodes"""

    def __init__(self):
        """initialize the class"""
        pass

def receive_args():
    """receive arguments from the user pass to the script"""
    parser = argparse.ArgumentParser(description='Pass arguments to the script')
    parser.add_argument('--csv_file',
                        dest='csv_file',
                        action='store',
                        choices=["iris.csv", "pima.csv"],
                        required=True)

    return parser

def main():
    """where the magic happens"""
    args = receive_args().parse_args()

main()