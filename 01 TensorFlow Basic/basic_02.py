import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

add = a + b
#add = tf.add(a, b)
mul = a * b
#mul = tf.mul(a, b)

sess = tf.Session()
sess.run(tf.local_variables_initializer())

print(a)
print(b)
print(add)
print(mul)
print("addition with variables, result=%i" % sess.run(add, feed_dict={a: 2, b: 3}))
print("multiplication with variables, result=%i" % sess.run(mul, feed_dict={a: 2, b: 3}))


