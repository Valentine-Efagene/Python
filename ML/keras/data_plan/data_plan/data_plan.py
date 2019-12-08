from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import keras.utils
from keras.optimizers import SGD
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt

input_size = 1
batch_size = 100
hidden_neurons = 100
epochs = 450
n_each = 1000
plans_MB = np.array([1500, 2000, 3500, 6500, 11000, 25000]);
plans_template = np.array([0, 1, 2, 3, 4, 5]);
standard_deviations = np.array([50, 60, 300, 350, 300, 2500])
usage_temp = np.zeros((len(plans_template), n_each))

for n in range(len(plans_template)):
    usage_temp[n] = np.random.normal(plans_MB[n], standard_deviations[n], n_each)

usages = usage_temp[0]

for n in range(len(plans_template) - 1):
    usages = np.concatenate((usages, usage_temp[n + 1]))

plans = np.repeat(plans_template, n_each)

# Display
data = ((usages[:n_each], plans[:n_each]), 
        (usages[n_each + 1:n_each * 2], plans[n_each + 1:n_each * 2]), 
        (usages[n_each * 2 + 1: n_each * 3], plans[n_each * 2 + 1: n_each * 3]), 
        (usages[n_each * 3 + 1: n_each * 4], plans[n_each * 3 + 1: n_each * 4]), 
        (usages[n_each * 4 + 1: n_each * 5], plans[n_each * 4 + 1: n_each * 5]),
        (usages[n_each * 5 + 1: n_each * 6], plans[n_each * 5 + 1: n_each * 6]))
colors = ("red", "blue", "yellow", "green", "purple", "black")
groups = ("1500", "2000", "3500", "6500", "11000", "25000")
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show() #Uncomment to display

# Remove to train
exit(0)

# Randomize the data
X = np.vstack((usages, plans))
K = X.T
np.random.shuffle(K)

usages = np.ravel(K.T[0])
usages /= 2500 # Normalize the data. Took accuracy from 0.16 to 0.97
plans = np.ravel(K.T[1])

training_fraction = 0.7

X_train = usages[ : (int)(training_fraction * n_each * len(plans_template))]
Y_train = plans[ : (int)(training_fraction * n_each * len(plans_template))]
X_test = usages[(int)(training_fraction * n_each * len(plans_template)) : ]
Y_test = plans[(int)(training_fraction * n_each * len(plans_template)) : ]

X_train = X_train.reshape((int)(training_fraction * n_each * len(plans_template)), 1)
X_test = X_test.reshape((int)((1 - training_fraction) * n_each * len(plans_template)), 1)


#logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/"
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
# Later go to logs folder, and type "Tensorboard --logdir scalars/", then type the url  http://GHOST:6006 in a browser.

# To one-hot
classes = 6
Y_train = keras.utils.to_categorical(Y_train, classes)
Y_test = keras.utils.to_categorical(Y_test, classes)

#model = Sequential([ Dense(hidden_neurons, input_dim=input_size), Activation('sigmoid'), Dense(classes), Activation('softmax') ])

model = Sequential();
model.add(Dense(hidden_neurons, activation='sigmoid', input_dim=input_size))
model.add(Dense(classes, activation='softmax'))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=True)
# Or use optimizer='sdg' in model.compile to use default parameters

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          callbacks=[tensorboard_callback])

score = model.evaluate(X_test, Y_test, verbose=1) 
print('Test accuracy:', score[1])

model.save('model.h5')