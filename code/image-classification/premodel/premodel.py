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
    k_fold = 10
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
    #t_0 /= 2

    predicted = []
    correct_result = []
    result_prediction = []
    predicted_models = []
    for p in all_predictions:
        for image, groundtruth_label, result_pred, prediction, model_predicted in p:
            correct_result.append(groundtruth_label)
            predicted.append(prediction) #<- this is model index
            result_prediction.append(result_pred)
            predicted_models.append(model_predicted)#<- this is model name
    time_elapsed = f"Time elapsed: {int(t_0 * 100) / 100} seconds"
    #res = (
    #    np.sum(np.array(predicted) == np.array(correct_result)) / len(predicted), list_premodels[counter], time_elapsed)

    #############################################################################################################################
    #Optimal model selector - prediction has to agree with nn (all nn's predict the same on val data - can use first level data)#
    #res = (np.sum(np.array(predicted) == np.array(correct_result)) / len(predicted), list_premodels[counter], time_elapsed)
    lookup = {"tf-mobilenet_v1":0.1, "tf-inception_v4":0.2, "tf-resnet_v1_152":0.6, "failed":0.0}
    time_with_nn = sum([lookup[x] for x in predicted_models])
    time_best_nn = len(predicted_models) * 0.6
    improvement = f"Best_nn: {time_best_nn} seconds,  premodel+nn time: {time_with_nn + t_0}, seconds"
    acc = 0
    n_correct = 0
    predicted_correct = 0
    for i, (gt, pred, res, model) in enumerate(zip(correct_result, predicted, result_prediction, predicted_models)):
        if int(first_level_data[i][1]) == gt: #oracle predictor
            n_correct += 1
            if gt == res:
                predicted_correct += 1
    #np.sum([int(first_level_data[x][1]) == int(result_prediction[x]) for x in range(len(first_level_data))]) / len(predicted)
    res = (predicted_correct/n_correct, list_premodels[counter], time_elapsed, improvement, predicted_correct/len(predicted_models))

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
    import matplotlib.pyplot as plt

    # plt.plot(a)
    # plt.show()
    # quit()
    # lets put it all together
    print("Starting...")
    models = machines_avialable()
    print("Models available: ")
    print(models)

    # amount_images = int(input("How many images do you want to use? "))
    amount_images = 10000
    # list_premodels = input("Which models do you want to use? ")
    list_premodels = [('knn', 'knn', 'knn'), ('dt16', 'dt16', 'dt16'), ('svc', 'svc', 'svc')]

    # percetange_results = prototype(amount_images, list_premodels)
    saveImg = True
    percetange_results, best_premodel = prototype(amount_images,
                                                  [[y for y in x.split() if y != "-"] for x in models])

    print("Results: ")
    print(percetange_results)
    #percetange_results.sort(key=lambda x: float(x[2].split()[2]))
    percetange_results.sort(key=lambda x: x[0])
    if amount_images == 50000:
        with open("output.txt", "a") as f:
            print("\n", file=f)
            print(percetange_results, file=f)
            print(percetange_results)
    times = [float(x[2].split()[2]) for x in percetange_results]
    times2 = [float(x[3].split()[5][:-1]) for x in percetange_results]
    print(times2)
    fig, host = plt.subplots(figsize=(8, 5), layout="constrained")
    ax2 = host.twinx()
    ax2.set_ylim(0.0, 0.9)
    host.set_ylim(0, 1.1)
    #host.bar(range(len(percetange_results)), [x / np.max(times) for x in times], label="Relative prediction time")
    host.bar(range(len(percetange_results)), [x / np.max(times2) for x in times2], label="Relative prediction time")
    p1 = host.plot(0, 0, label="Relative prediction time", color="b")
    host.set_xticks(range(len(percetange_results)), [str(x[1]) for x in percetange_results], rotation=90)
    maxval = max(times2)/amount_images
    host.set_yticks([0,0.2, 0.4, 0.6, 0.8, 1], [str(maxval*(i/5))[:4] for i in range(6)], rotation=90)
    p2 = ax2.plot(range(len(percetange_results)), [x[0] for x in percetange_results], ".-", color="r", label="Accuracy compared to Oracle")
    host.set_ylabel("Relative prediction time (s)")
    ax2.set_ylabel("Accuracy")
    #host.set_yticks([0])
    #ax2.legend()
    ax2.grid()
    ax2.legend(handles=p1+p2)
    plt.tight_layout()
    if saveImg:
        plt.savefig(f"premodel_orderedByAcc_{amount_images}.png")
    plt.show()
    print("Finished")
