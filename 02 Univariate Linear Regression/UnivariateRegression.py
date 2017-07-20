import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
#np.set_printoptions(threshold='nan')

#t_x = tf.floor(10 * np.random.random([5]))
#t_x = tf.floor(10 * tf.random_normal([5]))
X = 10 * np.random.random([5])
Y = X * 3.0 + 8.0
print("X: %s" % X)
print("Y: %s" % Y)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.Variable(0.0)
b = tf.Variable(0.0)
curr_y = a * x + b
loss = tf.reduce_sum(tf.square(curr_y - y))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    sess.run(train, feed_dict={x: X, y: Y})
    print(sess.run([a, b, loss], feed_dict={x: X, y: Y}))

exit(0)