import tensorflow as tf
x = tf.constant([[1, 220, 55], [4, 3, -1]])
x_max = tf.reduce_max(x, [0])
with tf.Session() as sess:
 print(sess.run(x_max))
