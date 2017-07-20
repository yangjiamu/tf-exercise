import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

hello = tf.constant('Hello,TensorFlow!')
sess = tf.Session()
print(sess.run(hello))