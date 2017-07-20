import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

a = tf.constant(2, name="a_const")
b = tf.constant(3, name="b_const")
c = a + b;

sess = tf.Session()
sess.run(tf.local_variables_initializer())

print(a)
print(b)
print(sess.run(a))
print(sess.run(b))

print(c)
print(sess.run(c))
