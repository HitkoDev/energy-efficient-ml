import tensorflow as tf
import tensorflow_hub as hub

## Fetch pretrained models from the paper (all models trained on ILSVRC-2012-CLS)

mobilenet_v1_25 = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v1/frameworks/TensorFlow2/variations/025-224-classification/versions/2")
])
mobilenet_v1_25.build([None, 224, 224, 3])

resnet_v1_50 = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/resnet-v1/frameworks/TensorFlow2/variations/152-classification/versions/2")
])
resnet_v1_50.build([None, 224, 224, 3])

inception_v2 = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/inception-v2/frameworks/TensorFlow2/variations/classification/versions/2")
])
inception_v2.build([None, 224, 224, 3])

resnet_v2_152 = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/152-classification/versions/2")
])
resnet_v2_152.build([None, 224, 224, 3])
