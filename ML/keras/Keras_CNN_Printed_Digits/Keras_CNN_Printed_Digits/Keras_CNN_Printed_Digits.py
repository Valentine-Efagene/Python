import cv2
import random
import numpy as np
from imageToArrayConverter import ImageToArrayConverter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from os import path
from keras.models import load_model

def pause():
    while cv2.waitKey() != ord('q'):
        return

if (not path.exists('model.h5')):
    itacTest = ImageToArrayConverter('printedDigitsTrain.png')
    itacTrain = ImageToArrayConverter('printedDigitsTrain.png')

    x_train = itacTrain.getImageAsArray()
    x_test = itacTest.getImageAsArray()

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

    y_train = np.array([6, 7, 4, 9, 5, 3, 1, 0, 8, 2])
    y_test = np.array([6, 7, 4, 9, 5, 3, 1, 0, 8, 2])
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
    model.fit(x=x_train,y=y_train, epochs=10)

    # EVALUATING THE MODEL
    model.evaluate(x_test, y_test)

    # Train with my image, and overfit by using train set as test set because I don't have much data
    #model.fit(x=x_vals_train,y=y_vals_train, epochs=1)
    #model.evaluate(x_vals_train, y_vals_train)

    model.save('model.h5')
else:
    model = load_model('model.h5')

itacTest = ImageToArrayConverter('try.png')

x_test = itacTest.getImageAsArray()

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#cv2.imshow('dhs', x_test[0])
#cv2.imshow('dhs', itacTest.getThresholdedImage())
#cv2.waitKey(0)
#exit(0)

x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_test /= 255

# Individual test
image_index = 10
img_rows = 28
img_cols = 28
cv2.imshow('f', x_test[image_index].reshape(28, 28))
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
#plt.show()

pause()
