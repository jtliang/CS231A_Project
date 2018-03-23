from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sqlite3
from random import random
from PIL import Image, ImageFilter
import os
import cv2

import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras import optimizers
# from sklearn.decomposition import PCA

num_features = 128

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 500, 500, 1]
  # Output Tensor Shape: [batch_size, 500, 500, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=32,
    kernel_size=[4, 4],
    padding="same",
    activation=tf.nn.relu)

  # Pooling Layer #1
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 6, 2, 64]
  # Output Tensor Shape: [batch_size, 6 * 2 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 32])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 6 * 2 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=99)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
  train_datas = []
  train_labels = []
  test_datas = []
  test_labels = []

  training = open("train.txt")
  train_datas = training.readline().rstrip().split(',')
  train_labels = training.readline().rstrip().split(',')
  testing = open("test.txt")
  test_datas = testing.readline().rstrip().split(',')
  test_labels = testing.readline().rstrip().split(',')

  train_data = []
  test_data = []

  maxs = 0

  for t in train_datas:
    image_file = Image.open("images/" + str(t) + ".jpg")
    image_file = image_file.convert('1')
    # width, height = image_file.size
    # maxwh = width if width > height else height
    # scale = 1633 // maxwh
    # image_file = image_file.resize((image_file.size[0] * scale, image_file.size[1] * scale))
    bw_im = Image.new('1', (1633, 1633), 0)
    arr = np.array(image_file)
    targ_x = int(1633/2 - arr.shape[1]/2)
    targ_y = int(1633/2 - arr.shape[0]/2)
    bw_im.paste(image_file, (targ_x, targ_y))
    bw_im = bw_im.resize((100, 100))
    bw_im = bw_im.filter(ImageFilter.FIND_EDGES)
    bw_im = np.array(bw_im)
    train_data.append(bw_im)

  for t in test_datas:
    image_file = Image.open("images/" + str(t) + ".jpg")
    image_file = image_file.convert('1')
    # width, height = image_file.size
    # maxwh = width if width > height else height
    # scale = 1633 // maxwh
    # image_file = image_file.resize((image_file.size[0] * scale, image_file.size[1] * scale))

    bw_im = Image.new('1', (1633, 1633), 0)
    arr = np.array(image_file)
    targ_x = int(1633/2 - arr.shape[1]/2)
    targ_y = int(1633/2 - arr.shape[0]/2)
    bw_im.paste(image_file, (targ_x, targ_y))
    bw_im = bw_im.resize((100, 100))
    bw_im = bw_im.filter(ImageFilter.FIND_EDGES)
    bw_im = np.array(bw_im)
    test_data.append(bw_im)

  train_labels = np.array(train_labels).astype(dtype='int64')
  test_labels = np.array(test_labels).astype(dtype='int64')
  train_data = np.array(train_data).astype(dtype='float32')
  test_data = np.array(test_data).astype(dtype='float32')

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=3000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  count = 0
  eval_results = mnist_classifier.predict(input_fn=eval_input_fn)
  for i, res in enumerate(eval_results):
    top3 = np.argsort(res['probabilities'])[-3:]
    if test_labels[i] not in top3:
      count += 1
  print((len(test_labels) - count) / len(test_labels))

if __name__ == "__main__":
  print("Starting Model Training")
  main()
