import tensorflow as tf

tf.reset_default_graph()

# Create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

# Launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('graphs', sess.graph)
    print(sess.run(c))
