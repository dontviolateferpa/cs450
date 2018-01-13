from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

train_test_split(iris.data, test_size=0.3, train_size=0.7)