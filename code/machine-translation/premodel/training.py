import math
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(n_estimators=10, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

meta_file = f"{os.path.dirname(__file__)}/data/premodels.tok.en.tok.csv"
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

    locs = [train[train["best_model"] == f] for f in train["best_model"].unique()]
    mx = max(v.shape[0] for v in locs)
    for v in locs:
        for z in range(math.floor(mx / v.shape[0]) - 1):
            train = pd.concat([train, v])

    test = pd.read_csv(test_file)

    le = preprocessing.LabelEncoder()
    le.fit(pd.concat([train["best_model"], test["best_model"]]))

    X_train = train.loc[
        :,
        (train.columns != "sentence")
        & (train.columns != "best_model")
        & (train.columns != "n_bpe_chars")
        & (train.columns != "n_tokens"),
    ]
    filter_col = [
        col
        for col in X_train
        if col.startswith("bow_") or col == "n_words" or col == "avg_adj"
    ]
    # X_train = X_train[filter_col].to_numpy(dtype=np.float32)
    y_train = le.transform(train["best_model"])

    X_test = test.loc[
        :,
        (test.columns != "sentence")
        & (test.columns != "best_model")
        & (test.columns != "n_bpe_chars")
        & (test.columns != "n_tokens"),
    ]
    filter_col = [
        col
        for col in X_test
        if col.startswith("bow_") or col == "n_words" or col == "avg_adj"
    ]
    # X_test = X_test[filter_col].to_numpy(dtype=np.float32)
    y_test = le.transform(test["best_model"])

    def score(pred, label):
        pred = le.inverse_transform(pred)
        bleu = 0
        rouge = 0
        dur = 0
        for s, p in zip(test["sentence"], pred):
            bleu += meta[f"{p}_bleu"][s]
            rouge += meta[f"{p}_rouge"][s]
            dur += meta[f"{p}_dur"][s]
        bleu /= len(pred)
        rouge /= len(pred)
        dur /= len(pred)
        if label not in scores:
            scores[label] = {"bleu": 0, "rouge": 0, "dur": 0}
        scores[label]["bleu"] += bleu
        scores[label]["rouge"] += rouge
        scores[label]["dur"] += dur
        return bleu, rouge, dur

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

    for name, clf in zip(names, classifiers):
        # clf = make_pipeline(PCA(), StandardScaler(), clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score(y_pred, name)
        print(
            "%s: Number of mislabeled points out of a total %d points : %d"
            % (name, X_test.shape[0], (y_test != y_pred).sum())
        )

    """nb = MultinomialNB()
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
    )"""

    scm = {}
    for k in scores:
        b = scores[k]["bleu"] / (i + 1)
        r = scores[k]["rouge"] / (i + 1)
        d = scores[k]["dur"] / (i + 1)
        scores[k]["f1"] = 2 * b * r / (b + r)
        scm[k] = {"bleu": b, "rouge": r, "dur": d, "f1": 2 * b * r / (b + r)}

    print(scm)
