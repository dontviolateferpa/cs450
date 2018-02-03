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

1. Group all target classes under an attribute value

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