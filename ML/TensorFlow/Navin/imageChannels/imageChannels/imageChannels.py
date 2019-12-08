import tensorflow as tf

tf.reset_default_graph()
image = tf.image.decode_png(tf.read_file("owl.png"), channels=3)

sess =  tf.InteractiveSession()
writer = tf.summary.FileWriter('graphs', sess.graph)
print(sess.run(tf.shape(image)))
