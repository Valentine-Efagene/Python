import cv2
import random
import numpy as np
from imageToArrayConverter import ImageToArrayConverter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from os import path
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

def pause():
    while cv2.waitKey() != ord('q'):
        return

if (not path.exists('model.h5')):
    itacTest = ImageToArrayConverter('digits.png')
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_test = itacTest.getImageAsArray()
    #y_test = np.array([0, 8, 9, 4, 5, 7, 1, 2, 6, 3])

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Display a data point (digit) to manually label the test set
    #cv2.imshow('Test', itacTest.getContourImage())
    #plt.imshow(np.reshape(x_test[9], (28, 28)), cmap='Greys')
    #plt.show()
    #pause()
    #exit(0)

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
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    loss = 'categorical_crossentropy'
    optimizer = 'adam'
    batch_size = 256
    epochs = 2

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # Train with my image, and overfit by using train set as test set because I don't have much data
    #model.fit(x=x_vals_train,y=y_vals_train, epochs=1)
    #model.evaluate(x_vals_train, y_vals_train)

    model.save('model.h5')
    print(score)
else:
    model = load_model('model.h5')

itacTest = ImageToArrayConverter('digits.png')
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = itacTest.getImageAsArray()

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#print(len(x_test))
#cv2.imshow('dhs', x_test[0])
#cv2.imshow('dhs', itacTest.getContourImage())
#pause()

x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_test /= 255

# Individual test
image_index = 7
img_rows = 28
img_cols = 28
cv2.imshow('f', x_test[image_index].reshape(img_rows, img_cols))
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
#plt.show()
pause()
