import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([x_data.shape[1], 1]))
b = tf.Variable(tf.random_normal([1]))

model = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean((-1)*Y*tf.log(model) + (-1)*(1-Y)*tf.log(1-model))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

prediction = tf.cast(model > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Training
	for step in range(100001):
		c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
		print(step, c)
	# Testing
	h, c, a = sess.run([model, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
	print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


