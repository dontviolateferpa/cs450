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

def main():
    """where the magic happens"""
    args = receive_args().parse_args()
    check_args(args)

main()