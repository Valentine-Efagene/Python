import cv2
import random
import numpy as np
from imageToArrayConverter import ImageToArrayConverter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from keras import callbacks
from os import path
from datetime import datetime
from keras.models import load_model
ops.reset_default_graph()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

def pause():
    while cv2.waitKey() != ord('q'):
        return

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

itacTest = ImageToArrayConverter('printedDigitsTrain.png')
#itacTrain = ImageToArrayConverter('printedDigitsTrain.png')
# Create graph
#sess = tf.Session()

x_vals_test = itacTest.getImageAsArray()

x_vals_test = x_vals_test.reshape(x_vals_test.shape[0], 28, 28, 1)

x_vals_test = x_vals_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_vals_test /= 255

print(x_vals_test.shape)
#cv2.imshow('k', itacTrain.getThresholdedImage())
#print(x_vals_train[8])


if (not path.exists('model.h5')):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #image_index = 3 # You may select anything up to 60,000
    #print(y_train[image_index]) # The label is 8
    #print(x_train[image_index])

    # Display a data point (digit)
    #plt.imshow(x_train[image_index], cmap='Greys')
    #plt.show()

    # Reshaping and Normalizing the Images
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    #print(x_train[image_index])
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    #print('Number of images in x_test', x_test.shape[0])

    # Building the Convolutional Neural Network
    # Importing the required Keras modules containing model and layers
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    # Creating a Sequential Model and adding the layers

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    # COMPILING AND FITTING THE MODEL
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(x=x_train,
              y=y_train,
              epochs=10,
              validation_data=(x_test, y_test), # I only added this for the callback thing from tf website with the next line
              callbacks=[tensorboard_callback]) # Added this and the line above when I was learning about tensorboard graphs for keras on tf website
    # Use tensorboard --logdir logs to visualise in powershell, while in the parent directory of logs.

    # EVALUATING THE MODEL
    model.evaluate(x_test, y_test)

    # Train with my image, and overfit by using train set as test set because I don't have much data
    #model.fit(x=x_vals_train,y=y_vals_train, epochs=1)
    #model.evaluate(x_vals_train, y_vals_train)

    model.save('model.h5')
else:
    model = load_model('model.h5')

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Individual test
image_index = 3
img_rows = 28
img_cols = 28
cv2.imshow('f', x_vals_test[image_index].reshape(28, 28))
#plt.imshow(x_vals_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_vals_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
#plt.show()

pause()