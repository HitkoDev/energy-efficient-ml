import tensorflow as tf
import tensorflow_hub as hub
import time
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
import cv2
## Fetch pretrained models from the paper (all models trained on ILSVRC-2012-CLS)


def make_DT(X_train, X_test, Y_train):
    """
    Decision Tree function that returns the prediction of a list of images. This function allows different deepth levels: 2,5,8,12 and 16
    Args:
        X_train: List of images features used for training
        X_test: List of images results used for validate the trained images.
        Y_train: List of images features predicted
    """
    tree = DecisionTreeRegressor(max_depth=16)
    tree.fit(X_train, X_test)
    return tree.predict(Y_train)

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

    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(Y_train)

    return predicted

def KNN(X_train, X_test, Y_test):
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
    predicted = knn.predict(Y_test)
    return predicted

if __name__ == "__main__":

    batch = [1,8,32, 64]
    for b in batch:
        print("batch:", b)
        for model in range(4):
            if model == 0:
                print("mobilenetv1")
                model = tf.keras.Sequential([
                    hub.KerasLayer(
                        "https://www.kaggle.com/models/google/mobilenet-v1/frameworks/TensorFlow2/variations/025-224-classification/versions/2")
                ])
                model.build([None, 224, 224, 3])
            # TEST
            # mobilenet_v1_25(tf.ones((16, 224, 224, 3)))
            elif model == 4:
                model = tf.keras.Sequential([
                    hub.KerasLayer(
                        "https://www.kaggle.com/models/google/resnet-v1/frameworks/TensorFlow2/variations/152-classification/versions/2")
                ])
                model.build([None, 224, 224, 3])
            elif model == 2:
                print("inceptionv2")
                model = tf.keras.Sequential([
                    hub.KerasLayer(
                        "https://www.kaggle.com/models/google/inception-v2/frameworks/TensorFlow2/variations/classification/versions/2")
                ])
                model.build([None, 224, 224, 3])
            elif model == 3:
                print("resnet152")
                model = tf.keras.Sequential([
                    hub.KerasLayer(
                        "https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/152-classification/versions/2")
                ])
                model.build([None, 224, 224, 3])
            if type(model) != int:

                img = cv2.resize(cv2.imread("./val_1/ILSVRC2012_val_00000001.JPEG"), (224, 224), interpolation=cv2.INTER_LINEAR)
                img = np.tile(img[None, :, :, :], (b, 1, 1, 1))
                t1 = time.time()
                model(img)
                t2 = time.time()
                print((t2 - t1)/b, "seconds")
