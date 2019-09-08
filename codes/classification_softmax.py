import tensorflow as tf

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]] # vectors
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]] # one hot encoding
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
W = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))

model = tf.nn.softmax(tf.add(tf.matmul(X,W),b))
cost = tf.reduce_mean(tf.reduce_sum((-1)*Y*tf.log(model), axis=1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Training
	for step in range(2001):
		c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
		print(step, c)
	# Testing
	test1 = sess.run(model, feed_dict={X: [[1,11,7,9]]})
	print(test1, sess.run(tf.argmax(test1, 1)))


