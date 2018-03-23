import numpy as np
import tensorflow as tf
import sqlite3
from random import random
from PIL import Image, ImageFilter
import os
import cv2

import keras
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras import optimizers

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
    width, height = image_file.size
    maxwh = width if width > height else height
    scale = 1633 / float(maxwh)
    image_file = image_file.resize((int(image_file.size[0] * scale), int(image_file.size[1] * scale)))
    bw_im = Image.new('1', (1633, 1633), 0)
    arr = np.array(image_file)
    targ_x = int(1633/2 - arr.shape[1]/2)
    targ_y = int(1633/2 - arr.shape[0]/2)
    bw_im.paste(image_file, (targ_x, targ_y))
    bw_im = image_file.resize((100, 100))
    bw_im = np.array(bw_im).reshape((100, 100, 1))
    train_data.append(bw_im)

  for t in test_datas:
    image_file = Image.open("images/" + str(t) + ".jpg")
    image_file = image_file.convert('1')
    width, height = image_file.size
    maxwh = width if width > height else height
    scale = 1633 / float(maxwh)
    image_file = image_file.resize((int(image_file.size[0] * scale), int(image_file.size[1] * scale)))
    bw_im = Image.new('1', (1633, 1633), 0)
    arr = np.array(image_file)
    targ_x = int(1633/2 - arr.shape[1]/2)
    targ_y = int(1633/2 - arr.shape[0]/2)
    bw_im.paste(image_file, (targ_x, targ_y))
    bw_im = image_file.resize((100, 100))
    bw_im = np.array(bw_im).reshape((100, 100, 1))
    test_data.append(bw_im)

  train_labels = np.array(train_labels).astype(dtype='int64')
  test_labels = np.array(test_labels).astype(dtype='int64')
  train_data = np.array(train_data).astype(dtype='float32')
  test_data = np.array(test_data).astype(dtype='float32')

  model = Sequential([
      Conv2D(8, (4, 4), padding='same', data_format='channels_last', input_shape=(100, 100, 1), activation='relu'),
      MaxPooling2D(pool_size=(2, 2), strides=2),
      Conv2D(32, (4, 4), padding='same', activation='relu'),
      MaxPooling2D(pool_size=(2, 2), strides=2),
      Flatten(),
      Dense(256, activation='relu', use_bias=True),
      Dropout(0.4),
      Dense(99, activation='softmax'),
  ])
  sgd = optimizers.SGD(lr=0.01)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  model.fit(train_data, train_labels, verbose=1, epochs=30, validation_split=0.1)
  evals = model.evaluate(test_data, test_labels, verbose=1)
  # print(evals)
  

  x = model.predict(test_data)
  count = 0
  for i, res in enumerate(x):
    top3 = np.argsort(res)[-3:]
    if test_labels[i] in top3:
      count += 1
    else:
      print(i, res, top3, test_data[i])
  # print((count) / float(len(test_labels)))

if __name__ == "__main__":
  print("Starting Model Training")
  main()