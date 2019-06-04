import tensorflow as tf
from classifier import *


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

tf.app.run(MNIST_classifier_estimator)
