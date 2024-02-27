import os

import pandas as pd

for i in range(10):
    # For each split, train & test the models
    train_file = (
        f"{os.path.dirname(__file__)}/data/premodels.tok.en_split_{i}_train.csv"
    )
    test_file = f"{os.path.dirname(__file__)}/data/premodels.tok.en_split_{i}_train.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    print(train)

    # TODO: train and compare the premodels
