import cv2
import random
import numpy as np
from imageToArrayConverter import ImageToArrayConverter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

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
y_vals_test = np.array([6, 7, 2, 8, 9, 5, 1, 3, 5])

x_vals_test = itacTest.getImageAsArray()
x_vals_train = itacTrain.getImageAsArray()
#cv2.imshow('f', np.reshape(x_vals_test[3], (28, 28)))
#cv2.imshow('f', itacTrain.getContourImage())
#cv2.waitKey(0)
#cv2.imshow('f', itacTrain.getThresholdedImage())
print(y_vals_train.shape)
print(x_vals_train.shape)
print(x_vals_test.shape)

n_inputs = 28*28
n_hidden1 = 10
n_hidden2 = 5
n_outputs = 10
learning_rate = 0.01
n_epochs = 400

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
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = x_vals_test # some new images (scaled from 0 to 1)
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    print(Z)
    print(y_pred)

#exit(0)
Nrows = 4
Ncols = 3
for i in range(itacTest.getNumberOfDigits()):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(x_vals_test[i], [28,28]), cmap='Greys_r')
    plt.title(' Pred: ' + str(y_vals_train[y_pred[i]]), fontsize=6)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    
plt.show()
