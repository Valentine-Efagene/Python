import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
features = 20
samples = 10000
classes = 10

x_train = np.random.random((samples, features))
y_train = keras.utils.to_categorical(np.random.randint(classes, size=(samples, 1)), num_classes=classes)
x_test = np.random.random((samples, features))
y_test = keras.utils.to_categorical(np.random.randint(classes, size=(samples, 1)), num_classes=classes)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=features))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# could also use optimizer='sdg' in model.compile to use default parameters
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
