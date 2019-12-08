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
predictors_vals_train, predictors_vals_test, target_vals_train, target_vals_test = train_test_split(predictors_vals, target_vals, test_size=0.2, random_state=12)

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables (Weight and Bias) that will be turned up
hidden_layer_nodes = 10
# For first layer
A1 = tf.Variable(tf.ones(shape=[1, hidden_layer_nodes])) # input -> idden nodes
b1 = tf.Variable(tf.ones(shape=[hidden_layer_nodes])) # ones biases for each hidden node

# For second layer
A2 = tf.Variable(tf.ones(shape=[hidden_layer_nodes, 1])) # input -> idden nodes
b2 = tf.Variable(tf.ones(shape=[1])) # ones biases for each hidden node

# Define Model Structure
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# Declare loss function (MSE) and optimizer
loss = tf.reduce_mean(tf.square(y_target - final_output))
my_opt = tf.train.AdamOptimizer(0.02) # Learning rate = 0.02
train_step = my_opt.minimize(loss)

# Initialize variables and session
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter('graphs', sess.graph)
sess.run(init)

# Training loop
lossArray = []
input = []
expected = []
output = []
input = []
test_loss = []
batch_size = 20

for i in range(500):
    batchIndex = np.random.choice(len(predictors_vals_train), size=batch_size)
    batchX = np.transpose( [predictors_vals_train[batchIndex]] )
    batchY = np.transpose( [target_vals_train[batchIndex]] )
    sess.run(train_step, feed_dict={x_data: batchX, y_target: batchY})

    batchLoss = sess.run(loss, feed_dict={x_data: batchX, y_target: batchY})
    lossArray.append(np.sqrt(batchLoss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: np.transpose([predictors_vals_test]),
                                              y_target: np.transpose([target_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))

    temp_output = sess.run(final_output, feed_dict={x_data: batchX, y_target: batchY})
    output.append(temp_output.ravel().tolist())

    expected.append(batchY.ravel().tolist())
    input.append(batchX.ravel().tolist())

    if(i + 1) % 50 == 0:
        print('Loss ' + str(batchLoss))

sess.close()

# output and expected are [[]] right now. outputToPlot will be []
outputToPlot = [item for sublist in output for item in sublist]
expectedToPlot = [item for sublist in expected for item in sublist]
inputToPlot = [item for sublist in input for item in sublist]

print('Expected ', batchY)
print('Output ', outputToPlot[:6])

# Plot loss over time
pyplot.plot(inputToPlot[:6], expectedToPlot[:6], 'r--',
           inputToPlot[:6], outputToPlot[:6])
#pyplot.plot(outputToPlot[:20], 'o-', label='Prediction')
#pyplot.plot(expectedToPlot[:20], 'r--', label='True vlaue')
pyplot.title('Loss per Generation')
pyplot.legend(loc='lower right')
#pyplot.xlabel('Generation')
#pyplot.ylabel('Loss')
pyplot.show()
