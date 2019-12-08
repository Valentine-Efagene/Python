import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from os import path
from keras.models import load_model
ops.reset_default_graph()

def pause():
    while cv2.waitKey() != ord('q'):
        return

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

y_vals_train = np.zeros( (10, 10) )
y_vals_train[0] = vectorized_result(6).ravel()
y_vals_train[1] = vectorized_result(7).ravel()
y_vals_train[2] = vectorized_result(2).ravel()
y_vals_train[3] = vectorized_result(8).ravel()
y_vals_train[4] = vectorized_result(9).ravel()
y_vals_train[5] = vectorized_result(5).ravel()
y_vals_train[6] = vectorized_result(1).ravel()
y_vals_train[7] = vectorized_result(3).ravel()
y_vals_train[8] = vectorized_result(6).ravel()
y_vals_train[9] = vectorized_result(4).ravel()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_test[0])

#x_vals_test = x_vals_test.reshape(x_vals_test.shape[0], 28, 28, 1)

#x_vals_test = x_vals_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
#x_vals_test /= 255

#print(x_vals_test.shape)
#cv2.imshow('k', itacTrain.getThresholdedImage())