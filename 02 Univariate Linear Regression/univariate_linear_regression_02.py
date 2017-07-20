import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.0001
num_steps = 100000

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = W * x + b
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(num_steps):
    sess.run(train, feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print("step %i: cost(%s) W(%s) b(%s)" %
              (step, str(sess.run(loss, feed_dict={x: x_data, y: y_data})),
               str(sess.run(W, feed_dict={x: x_data, y: y_data})),
               str(sess.run(b, feed_dict={x: x_data, y: y_data}))))

exit(0)
