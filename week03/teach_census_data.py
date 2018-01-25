"""
Prove Activity 02
"""
import pandas as pd
import numpy as np
from sklearn import datasets, model_selection
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


TARGETS = {
    b' <=50K': 0,
    b' >50K': 1
}

CONVERTERS = {
    14: lambda x: TARGETS[x]
}


COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country']

FILENAME = 'data\\adult.csv'

def prep_data():

    df = pd.io.parsers.read_csv(
        FILENAME,
        header=None,
        usecols=list(range(len(COLUMNS))),
        names=COLUMNS,
        na_values=[" ?"]
    )
    df["workclass"].fillna(df["workclass"].mode(), inplace=True)
    df.drop(columns='education', inplace=True)

    # df[['workclass', 'relationship']] = df[['workclass', 'relationship']].replace(['?'], [None])
    # df['age'] = df['age'].replace('?', np.NaN)

    # df.dropna(inplace=True)
    df["workclass"].fillna(df["workclass"].mode(), inplace=True)
    # print "work class mode"
    # print(df["workclass"].mode())
    df["occupation"].fillna(df["occupation"].mode(), inplace=True)

    print(df.isnull().sum())

    dummy_columns = [
        'workclass',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]

    prefixes = [
        'work',
        'maritial',
        'occupation',
        'rel',
        'race',
        'sex',
        'country'
    ]


    df = pd.get_dummies(df, columns=dummy_columns, prefix=prefixes)

    y = np.loadtxt(FILENAME, delimiter=',', skiprows=0, usecols=[14], converters=CONVERTERS)


    return model_selection.train_test_split(df, y, test_size=0.3, random_state=3)



def test_classifier(
        classifier,
        data_train,
        data_test,
        targets_train,
        targets_test
    ):
    """
    Test a model
    """
    model = classifier.fit(data_train, targets_train)

    targets_predicted = model.predict(data_test)
    print("Using classifier {}".format(classifier))
    print(targets_predicted)
    print(targets_test)
    print("{0:.2%} accurate\n".format(accuracy_score(targets_test, targets_predicted)))




def main():
    """The main function"""

    data = prep_data()

    classifiers = [GaussianNB()]

    for classifier in classifiers:
        test_classifier(classifier, *data)
    # print(data.head())


if __name__ == '__main__':
    main()

