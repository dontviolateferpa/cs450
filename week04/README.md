# Notes

## Instructions from Brother Burton

### 01

Self explanatory.

### 02

Just create a class.

### 03

You can do anything you want to prepare the data (open in excel and save as csv, etc.). Simply make sure the values are discretized.

### 04

Represent the tree! You can make a tree anyway you like (object-oriented, a dictionary of dictionaries, etc.).

### 05

?

### 06

?

### 07

?

### 08

?

## Pseudocode

### Find the entropy

1. Group all target classes under a feature value

- iterate through each column to do this

#### An example

| Row # | Credit Score | Income | Collateral | Should Loan |
|-------|--------------|--------|------------|-------------|
| 1     | Good         | High   | Good       | Yes         |
| 2     | Good         | High   | Poor       | Yes         |
| 3     | Good         | Low    | Good       | Yes         |
| 4     | Good         | Low    | Poor       | No          |
| 5     | Average      | High   | Good       | Yes         |
| 6     | Average      | Low    | Poor       | No          |
| 7     | Average      | High   | Poor       | Yes         |
| 8     | Average      | Low    | Good       | No          |
| 9     | Low          | High   | Good       | Yes         |
| 10    | Low          | High   | Poor       | No          |
| 11    | Low          | Low    | Good       | No          |
| 12    | Low          | Low    | Poor       | No          |

##### From Step 1

Iterate, get to `Credit Score`

Make a `Good` bucket

| Row # | Credit Score | Should Loan |
|-------|--------------|-------------|
| 1     | Good         | Yes         |
| 2     | Good         | Yes         |
| 3     | Good         | Yes         |
| 4     | Good         | No          |

Make an `Average` bucket

| Row # | Credit Score | Should Loan |
|-------|--------------|-------------|
| 5     | Average      | Yes         |
| 6     | Average      | No          |
| 7     | Average      | Yes         |
| 8     | Average      | No          |

Make a `Low` bucket

| Row # | Credit Score | Should Loan |
|-------|--------------|-------------|
| 9     | Low          | Yes         |
| 10    | Low          | No          |
| 11    | Low          | No          |
| 12    | Low          | No          |

### Build the Tree

```python
def build_node(data, feats)
    """
    this pseudocode assumes you have made a node class and that the children are represented
    inside the node as a dictionary
    """
    # BASE_CASE if no features left to split on (meaning you get to the end of the tree)
        # make new leaf node
        # use most common target for its value
        # return it
    # BASE CASE else if all rows in feature have the same target (entropy == 0?)
        # make new leaf node with that target
        # return it
    # BASE CASE no rows at all
        # ...
    # REAL LOGIC/MEAT: see how each feature/feature would split the data
        # choose the one with the lowest entropy
# possible_values = get possible values for feature
    for possible_value in possible_values:
        # data_subset = get rows from data where feature == possible_value, i.e. where gender=='M'
        data_subset = get_rows("logic in here")
        #### MAKE SURE TARGETS STAY LINED UP ####
        # node = build a node
        node = build_node("logic in here")
        current_node.children[possible_value] = node
    pass
```