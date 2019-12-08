import cv2
import random
import numpy as np
from imageToArrayConverter import ImageToArrayConverter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b

        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z

itacTest = ImageToArrayConverter('train.png')
itacTrain = ImageToArrayConverter('train.png')

y_vals_train = np.array([6, 7, 2, 8, 9, 5, 1, 3, 4, 0])
y_vals_test = np.array([5, 9, 2, 7, 3, 1, 0, 6, 4, 8])

x_vals_test = itacTest.getImageAsArray()
x_vals_train = itacTrain.getImageAsArray()
#cv2.imshow('f', np.reshape(x_vals_test[9], (28, 28)))
#cv2.imshow('f', itacTest.getContourImage())
#cv2.imshow('f', itacTrain.getThresholdedImage())
#cv2.waitKey(0)
print(y_vals_train.shape)
print(x_vals_train.shape)
print(x_vals_test.shape)

n_inputs = 28*28
n_hidden1 = 10
n_hidden2 = 5
n_outputs = 10
learning_rate = 0.01
n_epochs = 4009

# Create graph
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
Y = tf.placeholder(tf.int64, shape=(None), name="Y")

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(X, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=Y, logits=logits) # use softmax_cross_entropy_with_logits() for one-hot labels
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y_vals_train, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X:x_vals_train, Y:y_vals_train})
        acc_train = accuracy.eval(feed_dict={X:x_vals_train, Y:y_vals_train})
        acc_test = accuracy.eval(feed_dict={X:x_vals_test, Y:y_vals_test})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")






















'''y_vals_train = np.zeros( (10, 10) )
y_vals_train[0] = vectorized_result(6).ravel()
y_vals_train[1] = vectorized_result(7).ravel()
y_vals_train[2] = vectorized_result(2).ravel()
y_vals_train[3] = vectorized_result(8).ravel()
y_vals_train[4] = vectorized_result(9).ravel()
y_vals_train[5] = vectorized_result(5).ravel()
y_vals_train[6] = vectorized_result(1).ravel()
y_vals_train[7] = vectorized_result(3).ravel()
y_vals_train[8] = vectorized_result(6).ravel()
y_vals_train[9] = vectorized_result(4).ravel()'''