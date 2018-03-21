from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import difflib

COLS_CHESS  = ["wk_file", "wk_rank", "wr_file", "wr_rank", "bk_file", "bk_rank", "depth_of_win"]
COLS_IRIS   = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
COLS_LETTER = ["letter", "x-box", "y-box", "wid-wid", "high-height", "onpix", "x-bar", "y-bar",
               "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

np.set_printoptions(threshold=20)

def read_csv(cols, filename):
    return pd.io.parsers.read_csv(
        filename,
        header=None,
        usecols=list(range(len(cols))),
        names=cols
    )

def get_chess_dataset_categorical():
    df = read_csv(COLS_CHESS, "week11\\chess.csv")
    df_t = df["depth_of_win"]
    df.drop(columns=["depth_of_win"], inplace=True)

    return df, df_t

def get_letter_dataset_categorical():
    df = read_csv(COLS_LETTER, "week11\\letter.csv")
    # UNFINISHED
    return df

def get_iris_dataset_categorical():
    df = read_csv(COLS_IRIS, "week11\\iris.csv")
    df_t = df["class"]
    df.drop(columns=["class"], inplace=True)
    cols = df.columns
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

    df["sepal_len"] = pd.cut(df["sepal_len"], 3, labels=["Short", "Med", "Long"])
    df["sepal_wid"] = pd.cut(df["sepal_wid"], 3, labels=["Thin", "Med", "Thick"])
    df["petal_len"] = pd.cut(df["petal_len"], 3, labels=["Short", "Med", "Long"])
    df["petal_wid"] = pd.cut(df["petal_wid"], 3, labels=["Thin", "Med", "Thick"])

    return df, df_t

def get_chess_dataset_numerical():
    df = read_csv(COLS_CHESS, "week11\\chess.csv")
    df_t = df["depth_of_win"]
    df.drop(columns=["depth_of_win"], inplace=True)

    df["wk_file"] = df["wk_file"].astype('category').cat.codes
    df["wr_file"] = df["wr_file"].astype('category').cat.codes
    df["bk_file"] = df["bk_file"].astype('category').cat.codes

    df = df.apply(pd.to_numeric, args=('coerce',))

    return df, df_t

def get_letter_dataset_numerical():
    df = read_csv(COLS_LETTER, "week11\\letter.csv")
    # UNFINISHED
    return df

def get_iris_dataset_numerical():
    df = read_csv(COLS_IRIS, "week11\\iris.csv")
    df_t = df["class"]
    df.drop(columns=["class"], inplace=True)
    cols = df.columns
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df, df_t

def tts_chess_numeric():
    """train test split the numeric version of the chess dataset"""
    chess_num_d, chess_num_t= get_chess_dataset_numerical()
    return train_test_split(chess_num_d, chess_num_t, test_size=0.3, random_state=3)

def tts_chess_categorical():
    """train test split the categorical version of the chess dataset"""
    chess_cat_d, chess_cat_t= get_chess_dataset_categorical()
    return train_test_split(chess_cat_d, chess_cat_t, test_size=0.3, random_state=3)

def tts_iris_numeric():
    """train test split the numeric version of the iris dataset"""
    iris_num_d, iris_num_t= get_iris_dataset_numerical()
    return train_test_split(iris_num_d, iris_num_t, test_size=0.3, random_state=3)

def tts_iris_categorical():
    """train test split the categorical version of the iris dataset"""
    iris_cat_d, iris_cat_t= get_iris_dataset_categorical()
    return train_test_split(iris_cat_d, iris_cat_t, test_size=0.3, random_state=3)

def display_similarity(p, t, method):
    """display similarity between predictions (p) and test targets (t)"""
    if isinstance(p, pd.DataFrame) or isinstance(p, pd.Series):
        p = p.as_matrix()
    if isinstance(t, pd.DataFrame) or isinstance(t, pd.Series):
        t = t.as_matrix()
    error = np.mean(p != t)
    print("The two are " + str(1.0 - error) + " percent similar (" + method + ")")

def main():
    """magic happens here"""
    # preprocess, then train, test, and split
    chess_num_datatrain, chess_num_datatest, chess_num_targettrain, chess_num_targettest = tts_chess_numeric()
    chess_cat_datatrain, chess_cat_datatest, chess_cat_targettrain, chess_cat_targettest = tts_chess_categorical()
    iris_num_datatrain, iris_num_datatest, iris_num_targettrain, iris_num_targettest = tts_iris_numeric()
    iris_cat_datatrain, iris_cat_datatest, iris_cat_targettrain, iris_cat_targettest = tts_iris_categorical()
    
    # For each dataset
    ## Try at least 3 different "regular" learning algorithms and note the results.
    ### DS1 - chess
    print("")
    ##### method 1 - MLP **
    clf_chess_num_MLP = MLPClassifier(solver='adam', alpha=1e-5,
                                      hidden_layer_sizes=(40,30), random_state=1)
    clf_chess_num_MLP.fit(chess_num_datatrain, chess_num_targettrain)
    predictions = clf_chess_num_MLP.predict(chess_num_datatest)
    display_similarity(predictions, chess_num_targettest, "Chess - Neural Network")
    ##### method 2 - Decision Tree
    clf_chess_num_DT = DecisionTreeClassifier(random_state=0)
    clf_chess_num_DT.fit(chess_num_datatrain, chess_num_targettrain)
    predictions = clf_chess_num_DT.predict(chess_num_datatest)
    display_similarity(predictions, chess_num_targettest, "Chess - Decision Tree")
    ##### method 3
    clf_chess_num_KN = KNeighborsClassifier(n_neighbors=7)
    clf_chess_num_KN.fit(chess_num_datatrain, chess_num_targettrain)
    predictions = clf_chess_num_KN.predict(chess_num_datatest)
    display_similarity(predictions, chess_num_targettest, "Chess - KNN")
    ### DS2 - iris
    print("")
    ##### method 1 - MLP
    clf_iris_num_MLP = MLPClassifier(solver='adam', alpha=1e-5,
                                     hidden_layer_sizes=(10,7), random_state=1)
    clf_iris_num_MLP.fit(iris_num_datatrain, iris_num_targettrain)
    predictions = clf_iris_num_MLP.predict(iris_num_datatest)
    display_similarity(predictions, iris_num_targettest, "Iris - Neural Network")
    # iris_param_grid = [
    #     {
    #         'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    #         'solver' : ['lbfgs', 'sgd', 'adam'],
    #         'hidden_layer_sizes': [
    #          (9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,10),(9,11),(9,12),
    #          (10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,10),(10,11),(10,12),
    #          (11,1),(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,10),(11,11),(11,12)
    #          ]
    #     }
    #    ]
    # grid_clf = GridSearchCV(MLPClassifier, iris_param_grid, cv=3,
    #                        scoring='accuracy')
    # grid_clf.fit(iris_num_datatrain, iris_num_targettrain)
    # print("the best parameters out of those chosen are: ")
    # print(grid_clf.best_params_)
    ##### method 2
    ##### method 3
    ### DS3
    print("")
    ##### method 1
    ##### method 2
    ##### method 3

    ## Use Bagging and note the results. (Play around with a few different options)
    ### DS1
    ### DS2
    ### DS3

    ## Use AdaBoost and note the results. (Play around with a few different options)
    ### DS1
    ### DS2
    ### DS3

    ## Use a random forest and note the results. (Play around with a few different options)
    ### DS1
    ### DS2
    ### DS3

main()