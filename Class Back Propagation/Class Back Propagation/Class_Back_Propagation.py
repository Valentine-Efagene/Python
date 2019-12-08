#Importing the necessary modules 
import tensorflow as tf 
import numpy as np
 
NUM_EPOCHS = 100

x = np.array([0.05, 0.1]).reshape([2, 1])
y = np.array([0.01, 0.99]).reshape([2, 1])

X = tf.Variable( x, name="X", dtype=tf.float32) 

#first layer 
#Number of neurons = 2
w_h = tf.Variable(np.array([[0.15, 0.2], [0.25, 0.30]]), dtype=tf.float32) 
b_h = tf.Variable(np.array([0.35, 0.35]).reshape([2, 1]), dtype=tf.float32) 
h = tf.nn.sigmoid(tf.matmul(w_h, X) + b_h) 

#output layer 
#Number of neurons = 2
w_o = tf.Variable(np.array([[0.4, 0.45], [0.50, 0.55]]), dtype=tf.float32) 
b_o = tf.Variable(np.array([0.60, 0.60]).reshape([2, 1]), dtype=tf.float32)
output = tf.nn.sigmoid(tf.matmul(w_o, h) + b_o)
e = tf.abs(y - output)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(e)
init = tf.global_variables_initializer()

def optimize():
    with tf.Session() as session:
        session.run(init)
        #print(session.run(h))
        print("\nstarting at", "INPUT WEIGHTS:\n", session.run(w_h), "\nOUTPUT WEIGHTS:\n", session.run(w_o), "\nINPUT BIAS:\n", session.run(b_h), "\nOUTPUT BIAS:\n", session.run(b_o))

        for step in range(NUM_EPOCHS):
            session.run(train)
            print("\nstep", step, "INPUT WEIGHTS:\n", session.run(w_h), "\nOUTPUT WEIGHTS:\n", session.run(w_o), "\nINPUT BIAS:\n", session.run(b_h), "\nOUTPUT BIAS:\n", session.run(b_o))

        writer = tf.summary.FileWriter('graphs', session.graph)
        print('\nGraph saved\n')

        #print(session.run(output))

optimize()