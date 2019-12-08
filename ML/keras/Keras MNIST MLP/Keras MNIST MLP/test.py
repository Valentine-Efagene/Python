from os import path
from keras.models import load_model

if (not path.exists('model.h5')):
    print("There is no model to load!")
    exit(0)

model = load_model('model.h5')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
from keras.datasets import mnist
from keras.utils import np_utils


# To predict

image_index = 78

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test.reshape(10000, 784)
img_rows = 28
img_cols = 28
pred = model.predict(X_test[image_index].reshape(1, 784))
print(pred.argmax())

import cv2
cv2.imshow('f', X_test[image_index].reshape(28, 28))
cv2.waitKey(0)
exit(0)



fig = plt.figure()
weights = model.layers[0].get_weights()
w = weights[0].T
hidden_neurons = 100

for neuron in range(hidden_neurons):
    ax = fig.add_subplot(10, 10, neuron + 1)
    ax.axis("off")
    ax.imshow(numpy.reshape(w[neuron], (28, 28)), cmap=cm.Greys_r)

plt.savefig("neuron_images.png", dpi=300)
plt.show()