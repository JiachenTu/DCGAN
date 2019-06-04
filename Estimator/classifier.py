import tensorflow as tf
import numpy as np
from model import *

_NUM_CLASSES = 10
_MODEL_DIR = 'model_name'
_NUM_CHANNELS = 1
_IMG_SIZE = 28
_LEARNING_RATE = 0.05
_NUM_EPOCHS = 20
_BATCH_SIZE = 2048

def MNIST_classifier_estimator(_):

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns a np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns a np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create a input function to train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=_BATCH_SIZE,
        num_epochs=1,
        shuffle=True)
# Create a input function to eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        y=eval_labels,
        batch_size=_BATCH_SIZE,
        num_epochs=1,
        shuffle=False)
# Create a estimator with model_fn
    image_classifier = tf.estimator.Estimator(model_fn=model_fn,
                       model_dir=_MODEL_DIR)
# Finally, train and evaluate the model after each epoch
    for _ in range(_NUM_EPOCHS):
      image_classifier.train(input_fn=train_input_fn)
      metrics = image_classifier.evaluate(input_fn=eval_input_fn)
