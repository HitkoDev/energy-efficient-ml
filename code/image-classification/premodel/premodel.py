from __future__ import division
# import sys
import os
import csv
import time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
# Import datasets, classifiers and performance metrics
from sklearn import svm
# from sklearn.metrics import compare
import matplotlib.pyplot as plt
from math import ceil
import util
from threading import Thread

IMG_SIZE = 228
# Percentage_TESTED = 10
# DATA_FILE = 'all_image_features.csv'
# DATA_FILE = 'all_image_features_norm.csv'
##DATA_FILE = 'top-1-few-features.csv'
DATA_FILE = 'all_new_features_hier_norm.csv'
DATA_FILE = "new_features.csv"
DATA_FILE = "ostalaKoda/new_features.csv"
# PATH_TO_FILES = '/images/val/images'

list_machines = (
    'knn',
    'dt16',
    'svc'
)


def machines_avialable():
    list_machines_available = []
    for first_level_machine in list_machines:
        for second_level_machine in list_machines:
            for third_level_machine in list_machines:
                list_machines_available.append(
                    str(first_level_machine) + " - " + str(second_level_machine) + " - " + str(third_level_machine))

    return list_machines_available


def compare_array(list1, list2):
    """
    It compares to arrays and returns how many times they are equal
    """
    success = 0
    for position, number in enumerate(list1):
        if number == list2[position]:
            success = success + 1

    return success, len(list1) - success


def cv_training_data(amount_images):
    """
    This functions returns the data that will be used for training and test different machine learning models.
    This information is collected from the file DATA_FILE.
    Return:
        data: Data for training the models
        data_result: data for validation
    """

    data = []
    first_level = []
    second_level = []
    third_level = []

    # Getting the images for training and testing
    row_count = 0
    with open(DATA_FILE, 'rb') as csvfile:
        lines = [line.decode('utf-8-sig') for line in csvfile]

        for row in csv.reader(lines):
            # Remove the headers of csv file
            if row_count == 0:
                row_count = row_count + 1
                continue

            data.append([float(q) for q in row[-7:]])
            first_level.append((row[0], float(row[1])))
            second_level.append((row[0], float(row[2])))
            third_level.append((row[0], float(row[3])))
            row_count = row_count + 1
            if row_count > amount_images:
                break

    return data, first_level, second_level, third_level


# Nearest Neighbours model - TRAINING and PREDICTION
def nearest_neighbour(X_train, X_test, Y_test):
    """
    Nearest neighbour function that returns the prediction of a list of images. With K = 5
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """
    # Create and fit a nearest-neighbor classifier
    knn = KNeighborsClassifier()

    knn.fit(X_train, X_test)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1,
                         n_neighbors=5, p=2, weights='uniform')

    # Prediction
    time_0 = time.time()
    predicted = knn.predict(Y_test)

    time_1 = time.time()
    return predicted, time_1 - time_0


# Decision tree of level 2, 5,8,12 and 16 - TRAINING and PREDICTION
def decision_tree(X_train, X_test, Y_train):
    """
    Decision Tree function that returns the prediction of a list of images. This function allows different deepth levels: 2,5,8,12 and 16
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """

    # Create tree
    # regr_2 = DecisionTreeRegressor(max_depth=2)
    # regr_5 = DecisionTreeRegressor(max_depth=5)
    # regr_8 = DecisionTreeRegressor(max_depth=8)
    # regr_12 = DecisionTreeRegressor(max_depth=12)
    regr_16 = DecisionTreeRegressor(max_depth=16)

    # Fit tree
    # regr_2.fit(X_train, X_test)
    # regr_5.fit(X_train, X_test)
    # regr_8.fit(X_train, X_test)
    # regr_12.fit(X_train, X_test)
    regr_16.fit(X_train, X_test)

    # Predict
    # predicted_level_2 = regr_2.predict(Y_train)
    # predicted_level_5 = regr_5.predict(Y_train)
    # predicted_level_8 = regr_8.predict(Y_train)
    # predicted_level_12 = regr_12.predict(Y_train)

    time_0 = time.time()
    predicted_level_16 = regr_16.predict(Y_train)
    time_1 = time.time()
    return None, None, None, None, predicted_level_16, time_1 - time_0


# A support vector classifier model - TRAINING and PREDICTION
def vecto_classifier(X_train, X_test, Y_train):
    """
    Vector Classification function that returns the prediction of a list of images. With gamma 0.001
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    # We learn the digits on the first half of the digits
    classifier.fit(X_train, X_test)
    time_0 = time.time()
    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(Y_train)

    time_1 = time.time()
    return predicted, time_1 - time_0


def CV_fold_worker(test_idx, train_idx, img_data, first_level, second_level, third_level, first_level_machine,
                   second_level_machine, third_level_machine, return_wrapper, i):
    """
    Worker function for each fold in CV. Trains a model with training data, tests with
    test_idx. Places the results as (image, prediction) tuples in return wrapper
    Args:
        test_idx: List if indexes where the test_data is
        train_idx: List if indexes where the train_data is
        img_data: all of the image data
        first_level: The names of the classes, respective to model return
        return_wrapper: The list to add all results
    """
    if i % 10 == 0:
        print("Calculating fold", int((i / 65000) * 100), "%")
    # Create a validation set which is 10% of the training_data
    X_train = np.array(util.list_split(img_data, train_idx, [0])[0])

    Y_train = np.array(util.list_split(img_data, test_idx, [0])[0])
    Y_test_first_level, _ = util.list_split(first_level, test_idx, [0])
    Y_test_second_level, _ = util.list_split(second_level, test_idx, [0])
    Y_test_third_level, _ = util.list_split(third_level, test_idx, [0])

    X_test_first_level, _ = util.list_split(first_level, train_idx, [0])
    X_test_second_level, _ = util.list_split(second_level, train_idx, [0])
    X_test_third_level, _ = util.list_split(third_level, train_idx, [0])

    X_val_first_level = np.array([X_test_first_level[i][1] for i in range(0, len(X_test_first_level))])
    # Y_val_first_level = [Y_test_first_level[i][1] for i in range(0, len(Y_test_first_level))]

    X_val_second_level = np.array([X_test_second_level[i][1] for i in range(0, len(X_test_second_level))])
    # Y_val_second_level = [Y_test_second_level[i][1] for i in range(0, len(Y_test_second_level))]

    X_val_third_level = np.array([X_test_third_level[i][1] for i in range(0, len(X_test_third_level))])
    # Y_val_third_level = [Y_test_third_level[i][1] for i in range(0, len(Y_test_third_level))]

    list_predictions = []
    Y_train_second_level = []
    Y_train_second_level_position = []
    Y_train_third_level = []
    Y_train_third_level_position = []
    timed = 0
    ##################################################################################################################
    # First Level of hierarchy [Mobilnet_v1]
    ##################################################################################################################
    if first_level_machine == 'knn':
        predicted, t = nearest_neighbour(X_train, X_val_first_level, Y_train)
        timed += t
    elif first_level_machine == 'dt16':
        predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16, t = decision_tree(
            X_train, X_val_first_level, Y_train)
        timed += t
        predicted = predicted_level_16
    elif first_level_machine == 'svc':
        predicted, t = vecto_classifier(X_train, X_val_first_level, Y_train)
        timed += t
    else:
        raise NotImplementedError

    for position, prediction in enumerate(predicted):
        if first_level_machine == 'dt16':
            if prediction > 0.5:
                if Y_test_first_level[position][1] == 1:
                    list_predictions.append((Y_test_first_level[position][0], 1, prediction, 1, 'tf-mobilenet_v1'))
                else:
                    list_predictions.append((Y_test_first_level[position][0], 0, prediction, 1, 'tf-mobilenet_v1'))
            else:
                Y_train_second_level.append(Y_train[position])
                Y_train_second_level_position.append(position)
        else:
            if prediction == 1:
                if Y_test_first_level[position][1] == 1:
                    list_predictions.append((Y_test_first_level[position][0], 1, prediction, 1, 'tf-mobilenet_v1'))
                else:
                    list_predictions.append((Y_test_first_level[position][0], 0, prediction, 1, 'tf-mobilenet_v1'))
            else:
                Y_train_second_level.append(Y_train[position])
                Y_train_second_level_position.append(position)

    # Not necessary to go to the next level
    if len(Y_train_second_level) == 0:
        return_wrapper.append(list_predictions)
        return timed
    ##################################################################################################################
    # Second Level of hierarchy [Inception_v4]
    ##################################################################################################################
    # predicted = nearest_neighbour(X_train, X_val_second_level, Y_train_second_level)
    # predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val, Y_train)
    # predicted = predicted_level_16
    # predicted = vecto_classifier(X_train, X_val, Y_train)

    if second_level_machine == 'knn':
        predicted, t = nearest_neighbour(X_train, X_val_second_level, Y_train_second_level)
        timed += t
    elif second_level_machine == 'dt16':
        predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16, t = decision_tree(
            X_train, X_val_second_level, Y_train_second_level)
        timed += t
        predicted = predicted_level_16
    elif second_level_machine == 'svc':
        predicted, t = vecto_classifier(X_train, X_val_second_level, Y_train_second_level)
        timed += t

    for position, prediction in enumerate(predicted):
        if second_level_machine == 'dt16':
            if prediction > 0.5:
                if Y_test_second_level[position][1] == 1:
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 2,
                                             prediction, 2, 'tf-inception_v4'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 0,
                                             prediction, 2, 'tf-inception_v4'))
            else:
                Y_train_third_level.append(Y_train_second_level[position])
                Y_train_third_level_position.append(Y_train_second_level_position[position])
        else:
            if prediction == 1:
                if Y_test_second_level[position][1] == 1:
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 2,
                                             prediction, 2, 'tf-inception_v4'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_second_level_position[position]][0], 0,
                                             prediction, 2, 'tf-inception_v4'))
            else:
                Y_train_third_level.append(Y_train_second_level[position])
                Y_train_third_level_position.append(Y_train_second_level_position[position])

    if len(Y_train_third_level) == 0:
        return_wrapper.append(list_predictions)
        return timed

    ##################################################################################################################
    # Third Level of hierarchy [Resnet_v1_152]
    ##################################################################################################################
    # predicted = nearest_neighbour(X_train, X_val_third_level, Y_train_third_level)
    # predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16 = decision_tree(X_train, X_val_third_level, Y_train_third_level)
    # predicted = predicted_level_16
    # predicted = vecto_classifier(X_train, X_val, Y_train)

    if third_level_machine == 'knn':
        predicted, t = nearest_neighbour(X_train, X_val_third_level, Y_train_third_level)
        timed += t
    elif third_level_machine == 'dt16':
        predicted_level_2, predicted_level_5, predicted_level_8, predicted_level_12, predicted_level_16, t = decision_tree(
            X_train, X_val_third_level, Y_train_third_level)
        timed += t
        predicted = predicted_level_16
    elif third_level_machine == 'svc':
        predicted, t = vecto_classifier(X_train, X_val_third_level, Y_train_third_level)
        timed += t

    for position, prediction in enumerate(predicted):
        if third_level_machine == 'dt16':
            if prediction > 0.5:
                if Y_test_third_level[position][1] == 1:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3,
                                             prediction, 3, 'tf-resnet_v1_152'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0,
                                             prediction, 3, 'tf-resnet_v1_152'))
            else:
                if Y_test_third_level[position][1] == 1:
                    list_predictions.append(
                        (Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 4, 'failed'))
                else:
                    list_predictions.append(
                        (Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 4, 'failed'))
        else:
            if prediction == 1:
                if Y_test_third_level[position][1] == 1:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 3,
                                             prediction, 3, 'tf-resnet_v1_152'))
                else:
                    list_predictions.append((Y_test_first_level[Y_train_third_level_position[position]][0], 0,
                                             prediction, 3, 'tf-resnet_v1_152'))
            else:
                if Y_test_third_level[position][1] == 1:
                    list_predictions.append(
                        (Y_test_first_level[Y_train_third_level_position[position]][0], 3, prediction, 4, 'failed'))
                else:
                    list_predictions.append(
                        (Y_test_first_level[Y_train_third_level_position[position]][0], 0, prediction, 4, 'failed'))

    return_wrapper.append(list_predictions)
    return timed


def process(data, first_level_data, second_level_data, third_level_data, list_premodels, comb):
    counter, (first_level_machine, second_level_machine, third_level_machine) = comb
    k_fold = 2
    all_predictions = list()
    chunk_size = int(ceil(len(data) / float(k_fold)))
    # Create a new thread for each fold
    t_0 = 0
    for i, (test_idx, train_idx) in enumerate(util.chunkise(list(range(len(data))), chunk_size)):
        return_wrapper = list()
        t_0 += CV_fold_worker(test_idx, train_idx, data, first_level_data, second_level_data, third_level_data,
                              first_level_machine, second_level_machine, third_level_machine, return_wrapper,
                              i + counter * len(data) // 2)
        all_predictions.extend(return_wrapper)
    t_0 /= 2

    predicted = []
    correct_result = []
    result_prediction = []
    for p in all_predictions:
        for image, groundtruth_label, result_pred, prediction, model_predicted in p:
            correct_result.append(groundtruth_label)
            predicted.append(prediction)
            result_prediction.append(result_pred)
    time_elapsed = f"Time elapsed: {int(t_0 * 100) / 100} seconds"
    #res = (
    #    np.sum(np.array(predicted) == np.array(correct_result)) / len(predicted), list_premodels[counter], time_elapsed)

    #############################################################################################################################
    #Optimal model selector - prediction has to agree with nn (all nn's predict the same on val data - can use first level data)#

    res = (
        np.sum([int(first_level_data[x][1]) == int(result_prediction[x]) for x in range(len(first_level_data))]) / len(predicted), list_premodels[counter], time_elapsed)

    return res


def prototype(amount_images, list_premodels):
    """
    Produce a .csv file with the fields <Image_filename, Ground truth model, predicted model>
    for every image in the train information set. We use k-fold cross validation, where k=10.
    """
    percetange_results = []

    if len(list_premodels) == 0:
        print("No premodels were selected!")
        return percetange_results
    if amount_images == 0:
        print("No images were selected!")
        return percetange_results

    # print("Creating training data...")
    from functools import partial
    data, first_level_data, second_level_data, third_level_data = cv_training_data(amount_images)
    pool = Pool(8)
    func = partial(process, data, first_level_data, second_level_data, third_level_data, list_premodels)
    h = enumerate(list_premodels)
    outputs = pool.map(func, h)
    pool.close()
    pool.join()
    # for counter, (first_level_machine, second_level_machine, third_level_machine) in enumerate(list_premodels):
    # Split training data in k-fold chunks
    # Minimum needs to be 2

    #    percetange_results.append()
    percetange_results = outputs
    return percetange_results, None


if __name__ == "__main__":
    # inference times, without other processes
    a = [(0.6693533870677414, ['dt16', 'dt16', 'dt16'], 'Time elapsed: 0.0 seconds'),
         (0.7003540070801416, ['dt16', 'dt16', 'knn'], 'Time elapsed: 0.05 seconds'),
         (0.6992539850797016, ['dt16', 'knn', 'dt16'], 'Time elapsed: 0.07 seconds'),
         (0.6994539890797816, ['dt16', 'knn', 'knn'], 'Time elapsed: 0.09 seconds'),
         (0.7029940598811977, ['knn', 'dt16', 'dt16'], 'Time elapsed: 0.6 seconds'),
         (0.7024340486809736, ['knn', 'dt16', 'knn'], 'Time elapsed: 0.62 seconds'),
         (0.7024740494809896, ['knn', 'knn', 'dt16'], 'Time elapsed: 0.67 seconds'),
         (0.6438928778575571, ['knn', 'knn', 'knn'], 'Time elapsed: 0.81 seconds'),
         (0.7075341506830136, ['dt16', 'knn', 'svc'], 'Time elapsed: 1.4 seconds'),
         (0.7109142182843657, ['knn', 'dt16', 'svc'], 'Time elapsed: 1.93 seconds'),
         (0.7071541430828616, ['dt16', 'dt16', 'svc'], 'Time elapsed: 6.54 seconds'),
         (0.7075141502830057, ['dt16', 'svc', 'svc'], 'Time elapsed: 6.72 seconds'),
         (0.7077741554831096, ['dt16', 'svc', 'knn'], 'Time elapsed: 6.89 seconds'),
         (0.7075141502830057, ['dt16', 'svc', 'dt16'], 'Time elapsed: 7.04 seconds'),
         (0.7108142162843257, ['knn', 'svc', 'dt16'], 'Time elapsed: 10.55 seconds'),
         (0.7105542110842217, ['knn', 'svc', 'svc'], 'Time elapsed: 10.68 seconds'),
         (0.7105542110842217, ['knn', 'svc', 'knn'], 'Time elapsed: 10.84 seconds'),
         (0.7106142122842457, ['knn', 'knn', 'svc'], 'Time elapsed: 11.12 seconds'),
         (0.7068741374827496, ['svc', 'svc', 'svc'], 'Time elapsed: 46.47 seconds'),
         (0.7076141522830457, ['svc', 'svc', 'dt16'], 'Time elapsed: 49.23 seconds'),
         (0.7074741494829897, ['svc', 'svc', 'knn'], 'Time elapsed: 51.05 seconds'),
         (0.7076141522830457, ['svc', 'dt16', 'svc'], 'Time elapsed: 51.35 seconds'),
         (0.7074341486829736, ['svc', 'knn', 'svc'], 'Time elapsed: 59.86 seconds'),
         (0.7075941518830376, ['svc', 'dt16', 'dt16'], 'Time elapsed: 63.79 seconds'),
         (0.7074341486829736, ['svc', 'knn', 'knn'], 'Time elapsed: 64.06 seconds'),
         (0.7076541530830617, ['svc', 'dt16', 'knn'], 'Time elapsed: 64.12 seconds'),
         (0.7076941538830777, ['svc', 'knn', 'dt16'], 'Time elapsed: 64.23 seconds')]
    import matplotlib.pyplot as plt

    print(a)
    # plt.plot(a)
    # plt.show()
    # quit()
    # lets put it all together
    print("Starting...")
    models = machines_avialable()
    print("Models available: ")
    print(models)

    # amount_images = int(input("How many images do you want to use? "))
    amount_images = 50000
    # list_premodels = input("Which models do you want to use? ")
    list_premodels = [('knn', 'knn', 'knn'), ('dt16', 'dt16', 'dt16'), ('svc', 'svc', 'svc')]

    # percetange_results = prototype(amount_images, list_premodels)
    useSavedResults = False
    if not useSavedResults:
        percetange_results, best_premodel = prototype(amount_images,
                                                      [[y for y in x.split() if y != "-"] for x in models])
    else:
        percetange_results = a
    print("Results: ")
    #percetange_results.sort(key=lambda x: float(x[2].split()[2]))
    percetange_results.sort(key=lambda x: x[0])
    if amount_images == 50000:
        with open("output.txt", "a") as f:
            print("\n", file=f)
            print(percetange_results, file=f)
            print(percetange_results)
    times = [float(x[2].split()[2]) for x in percetange_results]
    fig, host = plt.subplots(figsize=(8, 5), layout="constrained")
    ax2 = host.twinx()
    ax2.set_ylim(0.0, 0.9)
    host.set_ylim(0, 1.1)
    host.bar(range(len(percetange_results)), [x / np.max(times) for x in times], label="Relative prediction time")
    p1 = host.plot(0, 0, label="Relative prediction time", color="b")
    host.set_xticks(range(len(percetange_results)), [str(x[1]) for x in percetange_results], rotation=90)
    p2 = ax2.plot(range(len(percetange_results)), [x[0] for x in percetange_results], "o-", color="r", label="Accuracy")
    host.set_ylabel("Relative prediction time")
    ax2.set_ylabel("Accuracy")
    host.set_yticks([0])
    #ax2.legend()
    ax2.grid()
    ax2.legend(handles=p1+p2)
    plt.tight_layout()
    plt.show()
    print("Finished")
