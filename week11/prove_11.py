from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
    # UNFINISHED
    return df

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
    # UNFINISHED
    return df

def tts_chess_numeric():
    """train test split the numeric version of the chess dataset"""
    chess_num_d, chess_num_t= get_chess_dataset_numerical()
    return train_test_split(chess_num_d, chess_num_t, test_size=0.3, random_state=3)

def tts_chess_categorical():
    """train test split the numeric version of the chess dataset"""
    chess_cat_d, chess_cat_t= get_chess_dataset_categorical()
    return train_test_split(chess_cat_d, chess_cat_t, test_size=0.3, random_state=3)

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
    # preprocessing
    iris_num_d = get_iris_dataset_numerical()
    iris_cat_d = get_iris_dataset_categorical()

    chess_num_datatrain, chess_num_datatest, chess_num_targettrain, chess_num_targettest = tts_chess_numeric()
    chess_cat_datatrain, chess_cat_datatest, chess_cat_targettrain, chess_cat_targettest = tts_chess_categorical()
    # For each dataset

    ## Try at least 3 different "regular" learning algorithms and note the results.
    ### DS1 - chess
    ##### method 1 - MLP **
    clf_chess_num_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                  hidden_layer_sizes=(40,30), random_state=1)
    clf_chess_num_MLP.fit(chess_num_datatrain, chess_num_targettrain)
    predictions = clf_chess_num_MLP.predict(chess_num_datatest)
    display_similarity(predictions, chess_num_targettest, "Neural Network on Chess Dataset")
    ##### method 2 - Decision Tree
    clf_chess_num_DT = DecisionTreeClassifier(random_state=0)
    clf_chess_num_DT.fit(chess_num_datatrain, chess_num_targettrain)
    predictions = clf_chess_num_DT.predict(chess_num_datatest)
    display_similarity(predictions, chess_num_targettest, "Decision Tree on Chess Dataset")
    ##### method 3
    ### DS2
    ##### method 1
    ##### method 2
    ##### method 3
    ### DS3
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