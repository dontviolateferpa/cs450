from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
import difflib

COLS_CHESS  = ["wk_file", "wk_rank", "wr_file", "wr_rank", "bk_file", "bk_rank", "depth_of_win"]
COLS_IRIS   = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
COLS_LETTER = ["letter", "x-box", "y-box", "wid-wid", "high-height", "onpix", "x-bar", "y-bar",
               "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

np.set_printoptions(threshold=20)

def read_csv(cols, filename, dtype_p=None):
    return pd.io.parsers.read_csv(
        filename,
        header=None,
        usecols=list(range(len(cols))),
        names=cols,
        dtype=dtype_p
    )

def get_chess_dataset_categorical():
    df = read_csv(COLS_CHESS, "week11\\chess.csv")
    df_t = df["depth_of_win"]
    df.drop(columns=["depth_of_win"], inplace=True)

    return df, df_t

def get_letter_dataset_categorical():
    df = read_csv(COLS_LETTER, "week11\\letters.csv")
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
    cols_to_type = {'x-box': np.int64, 'y-box': np.int64, 'wid-wid': np.int64,
                    'high-height': np.int64, 'onpix': np.int64, 'x-bar': np.int64,
                    'y-bar': np.int64, 'x2bar': np.int64, 'y2bar': np.int64,
                    'xybar': np.int64, 'x2ybr': np.int64, 'xy2br': np.int64,
                    'x-ege': np.int64, 'xegvy': np.int64, 'y-ege': np.int64,
                    'yegvx': np.int64}
    df = read_csv(COLS_LETTER, "week11\\letters.csv", dtype_p=cols_to_type)
    df_t = df["letter"]
    df.drop(columns=["letter"], inplace=True)

    for col in df.columns:
        df[col] = df[col].astype(np.float64)

    return df, df_t

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

def tts_iris_numeric():
    """train test split the numeric version of the iris dataset"""
    iris_num_d, iris_num_t= get_iris_dataset_numerical()
    return train_test_split(iris_num_d, iris_num_t, test_size=0.3, random_state=3)

def tts_letter_numeric():
    """train test split the numeric version of the letter dataset"""
    letter_num_d, letter_num_t= get_letter_dataset_numerical()
    return train_test_split(letter_num_d, letter_num_t, test_size=0.3, random_state=3)


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
    iris_num_datatrain, iris_num_datatest, iris_num_targettrain, iris_num_targettest = tts_iris_numeric()
    letter_num_datatrain, letter_num_datatest, letter_num_targettrain, letter_num_targettest = tts_letter_numeric()
    
    # # For each dataset
    # ## Try at least 3 different "regular" learning algorithms and note the results.
    # ### DS1 - chess
    # print("")
    # ##### method 1 - MLP **
    # clf_chess_num_MLP = MLPClassifier(solver='adam', alpha=1e-5,
    #                                   hidden_layer_sizes=(40,30), random_state=1)
    # clf_chess_num_MLP.fit(chess_num_datatrain, chess_num_targettrain)
    # predictions = clf_chess_num_MLP.predict(chess_num_datatest)
    # display_similarity(predictions, chess_num_targettest, "Chess - Neural Network")
    # ##### method 2 - Decision Tree
    # clf_chess_num_DT = DecisionTreeClassifier(random_state=0)
    # clf_chess_num_DT.fit(chess_num_datatrain, chess_num_targettrain)
    # predictions = clf_chess_num_DT.predict(chess_num_datatest)
    # display_similarity(predictions, chess_num_targettest, "Chess - Decision Tree")
    # ##### method 3 - KNN
    # clf_chess_num_KNN = KNeighborsClassifier(n_neighbors=7)
    # clf_chess_num_KNN.fit(chess_num_datatrain, chess_num_targettrain)
    # predictions = clf_chess_num_KNN.predict(chess_num_datatest)
    # display_similarity(predictions, chess_num_targettest, "Chess - KNN")
    # ### DS2 - iris
    # print("")
    # ##### method 1 - MLP
    # clf_iris_num_MLP = MLPClassifier(solver='adam', alpha=1e-5,
    #                                  hidden_layer_sizes=(10,7), random_state=1)
    # clf_iris_num_MLP.fit(iris_num_datatrain, iris_num_targettrain)
    # predictions = clf_iris_num_MLP.predict(iris_num_datatest)
    # display_similarity(predictions, iris_num_targettest, "Iris - Neural Network")
    # # clf_iris_num_MLP_gs = MLPClassifier()
    # # iris_param_grid = [
    # #     {
    # #         'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    # #         'solver' : ['lbfgs', 'sgd', 'adam'],
    # #         'hidden_layer_sizes': [
    # #          (9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,10),(9,11),(9,12),
    # #          (10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,10),(10,11),(10,12),
    # #          (11,1),(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,10),(11,11),(11,12)
    # #          ]
    # #     }
    # #    ]
    # # grid_clf = GridSearchCV(clf_iris_num_MLP_gs, iris_param_grid, cv=3,
    # #                        scoring='accuracy')
    # # grid_clf.fit(iris_num_datatrain, iris_num_targettrain)
    # # print("the best parameters out of those chosen are: ")
    # # print(grid_clf.best_params_)
    # ##### method 2 - Decision Tree
    # clf_iris_num_DT = DecisionTreeClassifier()
    # clf_iris_num_DT.fit(iris_num_datatrain, iris_num_targettrain)
    # predictions = clf_iris_num_DT.predict(iris_num_datatest)
    # display_similarity(predictions, iris_num_targettest, "Iris - Decision Tree")
    # ##### method 3 - KNN
    # clf_iris_num_KNN = KNeighborsClassifier(n_neighbors=3)
    # clf_iris_num_KNN.fit(iris_num_datatrain, iris_num_targettrain)
    # predictions = clf_iris_num_KNN.predict(iris_num_datatest)
    # display_similarity(predictions, iris_num_targettest, "Iris - KNN")
    # ### DS3
    # print("")
    # ##### method 1 - MLP
    # clf_letter_num_MLP = MLPClassifier(solver='adam', alpha=1e-5,
    #                                   hidden_layer_sizes=(40,30), random_state=1)
    # clf_letter_num_MLP.fit(letter_num_datatrain, letter_num_targettrain)
    # predictions = clf_letter_num_MLP.predict(letter_num_datatest)
    # display_similarity(predictions, letter_num_targettest, "Letter - Neural Network")
    # ##### method 2 - Decision Tree
    # clf_letter_num_DT = DecisionTreeClassifier()
    # clf_letter_num_DT.fit(letter_num_datatrain, letter_num_targettrain)
    # predictions = clf_letter_num_DT.predict(letter_num_datatest)
    # display_similarity(predictions, letter_num_targettest, "Letter - Decision Tree")
    # ##### method 3 - KNN
    # clf_letter_num_KNN = KNeighborsClassifier(n_neighbors=3)
    # clf_letter_num_KNN.fit(letter_num_datatrain, letter_num_targettrain)
    # predictions = clf_letter_num_KNN.predict(letter_num_datatest)
    # display_similarity(predictions, letter_num_targettest, "Letter - KNN")

    ## Use Bagging and note the results. (Play around with a few different options)
    ### DS1 - Chess
    clf_chess_num_Bagging = BaggingClassifier(bootstrap=True, n_estimators=20)
    clf_chess_num_Bagging.fit(chess_num_datatrain, chess_num_targettrain)
    predictions = clf_chess_num_Bagging.predict(chess_num_datatest)
    display_similarity(predictions, chess_num_targettest, "BAGGING - Chess")
    ### DS2 - Iris
    clf_iris_num_Bagging = BaggingClassifier(bootstrap=True)
    clf_iris_num_Bagging.fit(iris_num_datatrain, iris_num_targettrain)
    predictions = clf_iris_num_Bagging.predict(iris_num_datatest)
    display_similarity(predictions, iris_num_targettest, "BAGGING - Iris")
    ### DS3 - Letter
    clf_letter_num_Bagging = BaggingClassifier(bootstrap=False)
    clf_letter_num_Bagging.fit(letter_num_datatrain, letter_num_targettrain)
    predictions = clf_letter_num_Bagging.predict(letter_num_datatest)
    display_similarity(predictions, letter_num_targettest, "BAGGING - Letter")

    ## Use AdaBoost and note the results. (Play around with a few different options)
    ### DS1 - Chess
    ### DS2 - Iris
    ### DS3 - Letter

    ## Use a random forest and note the results. (Play around with a few different options)
    ### DS1 - Chess
    ### DS2 - Iris
    ### DS3 - Letter

main()