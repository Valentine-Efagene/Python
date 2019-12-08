
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

iris = datasets.load_iris()
# X is Sepal.Length and Y is Petal Length
predictors_vals = np.array( [ predictors[0] for predictors in iris.data ] )
target_vals = np.array( [ predictors[2] for predictors in iris.data ] )

# Split data into train and test 80% - 20%
x_trn, x_tst, y_trn, y_tst = train_test_split(predictors_vals,
                                             target_vals, test_size=0.2, random_state=12)

predictor = tf.placeholder(shape=[None, 1], dtype=tf.float32)
target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables (weigths and bias) that can be tuned up
A = tf.Variable(tf.zeros(shape=[1, 1]))
b = tf.Variable(tf.ones(shape=[1, 1]))

# Declare model operation
model_output = tf.add(tf.matmul(predictor, A), b)

# Declare loss function and optimizer
loss = tf.reduce_mean(tf.abs(target - model_output))
my_opt = tf.train.GradientDescentOptimizer(0.01)
# my_opt = tf.train.AdamOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables and session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Fit model by using training loop
# Training loop
lossArray = []
batch_size = 40

for i in range(200):
    rand_rows = np.random.randint(0, len(x_trn) - 1, size=batch_size)
    batchX = np.transpose( [x_trn[rand_rows]] )
    batchY = np.transpose( [y_trn[rand_rows]] )
    sess.run(train_step, feed_dict={predictor: batchX, target: batchY})
    batchLoss = sess.run(loss, feed_dict={predictor: batchX, target: batchY})
    lossArray.append(batchLoss)

    if(i + 1) % 50 == 0:
        print('Step Number ' + str(i + 1) + 'A = ' + str(sess.run(A)) 
              + ' b = ' + str(sess.run(b)) )
        print('L1 loss = ' + str(batchLoss))

    [slope] = sess.run(A)
    [y_intercept] = sess.run(b)

    # Check and display the result on test data
    lossArray = []
    batch_size = 30

for i in range(100):
    rand_rows = np.random.randint(0, len(x_trn) - 1, size=batch_size)
    batchX = np.transpose( [x_trn[rand_rows]] )
    batchY = np.transpose( [y_trn[rand_rows]] )
    sess.run(train_step, feed_dict={predictor: batchX, target: batchY})
    batchLoss = sess.run(loss, feed_dict={predictor: batchX, target: batchY})
    lossArray.append(batchLoss)
        
    if(i + 1) % 20 == 0:
        print('Step Number ' + str(i + 1) + 'A = ' + str(sess.run(A)) 
              + ' b = ' + str(sess.run(b)) )
        print('L1 loss = ' + str(batchLoss))

# Get the optimal coefficients
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# Original Data and plot
plt.plot(x_tst, y_tst, 'o', label='Actual Data')
test_fit = []

for i in x_tst:
    test_fit.append(slope * i + y_intercept)

# Predicted values and plot
plt.plot(x_tst, test_fit, 'r-', label='Predicted line', linewidth=1)
plt.legend(loc='lower right')
plt.ylabel('Petal Length vs Sepal Length')
plt.xlabel('Petal Length')
plt.xlabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(lossArray, 'r-')
plt.title('L1 loss')
plt.show()