import os

import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

meta_file = f"{os.path.dirname(__file__)}/../translated/premodels.tok.csv"
meta = pd.read_csv(meta_file)

scores = {}
n = 10

for i in range(n):
    # For each split, train & test the models
    train_file = (
        f"{os.path.dirname(__file__)}/data/premodels.tok.en_split_{i}_train.csv"
    )
    test_file = f"{os.path.dirname(__file__)}/data/premodels.tok.en_split_{i}_test.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    X_train = train.loc[
        :, (train.columns != "sentence") & (train.columns != "best_model")
    ]
    y_train = train["best_model"]

    X_test = test.loc[:, (test.columns != "sentence") & (test.columns != "best_model")]
    y_test = test["best_model"]

    def score(pred, label):
        bleu = 0
        rouge = 0
        for s, p in zip(test["sentence"], pred):
            bleu += meta[f"{p}_bleu"][s]
            rouge += meta[f"{p}_rouge"][s]
        bleu /= len(pred)
        rouge /= len(pred)
        if label not in scores:
            scores[label] = {"bleu": 0, "rouge": 0}
        scores[label]["bleu"] += bleu
        scores[label]["rouge"] += rouge
        return bleu, rouge

    # TODO: train and compare the premodels
    """
    Premodels:
        feature stacking
        multi DT
        single DT
        multi KNN
        single KNN
        multi SVM
        single SVM
        multi NB
        single NB

    Reported metric:
        f1 score (low to high)
        total inference time
    """

    nb = MultinomialNB()
    y_pred = nb.fit(X_train, y_train).predict(X_test)
    score(y_pred, "single_NB")
    print(
        "NB: Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum())
    )

    sv = svm.SVC()
    y_pred = sv.fit(X_train, y_train).predict(X_test)
    score(y_pred, "single_SVM")
    print(
        "SVM: Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum())
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    y_pred = knn.fit(X_train, y_train).predict(X_test)
    score(y_pred, "single_KNN")
    print(
        "KNN: Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum())
    )

    dt = DecisionTreeClassifier(random_state=0)
    y_pred = dt.fit(X_train, y_train).predict(X_test)
    score(y_pred, "single_DT")
    print(
        "DT: Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum())
    )

print(scores)

for k in scores:
    b = scores[k]["bleu"] / n
    r = scores[k]["rouge"] / n
    print(f"{k}: {2*b*r/(b+r)}")
