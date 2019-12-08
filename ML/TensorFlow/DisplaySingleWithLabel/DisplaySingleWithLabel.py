# MNIST Digit Prediction with k-Nearest Neighbors
#-----------------------------------------------
#
# This script will load the MNIST data, and split
# it into test/train and perform prediction with
# nearest neighbors
#
# For each test integer, we will return the
# closest image/integer.
#
# Integer images are represented as 28x8 matrices
# of floating point numbers

import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Random sample
train_size = 1000
test_size = 102
train_data = mnist.train.images
test_data = mnist.test.images
train_labels = mnist.train.labels
test_labels = mnist.test.labels
rand_train_indices = np.random.choice(len(train_data), train_size, replace=False)
rand_test_indices = np.random.choice(len(test_data), test_size, replace=False)
x_vals_train = train_data[rand_train_indices]
x_vals_test = test_data[rand_test_indices]
y_vals_train = train_labels[rand_train_indices]
y_vals_test = test_labels[rand_test_indices]

i = np.random.choice(np.random.choice(len(train_data)))
print( train_labels[i] )
some_digit = train_data[i]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()