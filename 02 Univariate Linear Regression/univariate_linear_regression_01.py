import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.0001
num_steps = 100000

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(num_steps):
    sess.run(train)
    if step % 20 == 0:
        print("step %i: cost(%s) W(%s) b(%s)" % (step, str(sess.run(cost)), str(sess.run(W)), str(sess.run(b))))

exit(1)