import tensorflow as tf

x = tf.Variable(2, name='x', dtype=tf.float32)
y = tf.abs(tf.sin(x))
#log_x = tf.log(x)
#y = tf.square(log_x)
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(y)
init = tf.global_variables_initializer()

def optimize():
    with tf.Session() as session:
        session.run(init)
        print("starting at", "x:", session.run(x), "|sin(x)|:", session.run(y))

        for step in range(100):
            session.run(train)
            print("step", step, "x:", session.run(x), "|sin(x)|:", session.run(y))

        writer = tf.summary.FileWriter('graphs', session.graph)
        print('\nGraph saved\n')

optimize()