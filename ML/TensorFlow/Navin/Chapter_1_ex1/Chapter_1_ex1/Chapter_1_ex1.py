import tensorflow as tf
sess = tf.Session()

# Creating a new graph (not default)
myGraph = tf.Graph()

with myGraph.as_default():
    variable = tf.Variable(30, name="navin")
    initialize = tf.global_variables_initializer()

with tf.Session(graph=myGraph) as sess:
    sess.run(initialize)
    print(sess.run(variable))

import os

merged = tf.summary.merge_all()

if not os.path.exists('tensorboard_logs/'):
    os.mkdirs('tensorboard_logs/')

my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)