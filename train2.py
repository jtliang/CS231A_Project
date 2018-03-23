from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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
    bw_im = Image.new('RGB', (1633, 1633), (0, 0, 0))
    arr = np.array(image_file)
    targ_x = int(1633/2 - arr.shape[1]/2)
    targ_y = int(1633/2 - arr.shape[0]/2)
    bw_im.paste(image_file, (targ_x, targ_y))
    bw_im = bw_im.resize((224, 224))
    bw_im = bw_im.filter(ImageFilter.FIND_EDGES)
    bw_im = np.array(bw_im)
    train_data.append(bw_im)

  for t in test_datas:
    image_file = Image.open("images/" + str(t) + ".jpg")
    bw_im = Image.new('RGB', (1633, 1633), (0, 0, 0))
    arr = np.array(image_file)
    targ_x = int(1633/2 - arr.shape[1]/2)
    targ_y = int(1633/2 - arr.shape[0]/2)
    bw_im.paste(image_file, (targ_x, targ_y))
    bw_im = bw_im.resize((224, 224))
    bw_im = bw_im.filter(ImageFilter.FIND_EDGES)
    bw_im = np.array(bw_im)
    test_data.append(bw_im)

  train_labels = np.array(train_labels).astype(dtype='int64')
  test_labels = np.array(test_labels).astype(dtype='int64')
  train_data = np.array(train_data).astype(dtype='float32')
  test_data = np.array(test_data).astype(dtype='float32')

  test_labels = keras.utils.np_utils.to_categorical(test_labels, num_classes=99)
  train_labels = keras.utils.np_utils.to_categorical(train_labels, num_classes=99)

  vgg16_model = keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=99)
  sgd = optimizers.SGD(lr=0.01)
  vgg16_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  vgg16_model.fit(train_data, train_labels, verbose=1, epochs=25, validation_split=0.1)
  evals = vgg16_model.evaluate(test_data, test_labels, verbose=1)
  print(evals)
  print("\n\n")
  print(vgg16_model.metrics_names)

if __name__ == "__main__":
  print("Starting Model Training")
  main()
