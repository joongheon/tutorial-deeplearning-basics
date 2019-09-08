from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import time

training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
X_img = tf.reshape(X, [-1, 28, 28, 1])

# Convolution Layer 1
CL1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding='SAME', strides=1, activation=tf.nn.relu)
# Pooling Layer 1
PL1 = tf.layers.max_pooling2d(inputs=CL1, pool_size=[2,2], padding='SAME', strides=2)
# Convolution Layer 2
CL2 = tf.layers.conv2d(inputs=PL1, filters=64, kernel_size=[3,3], padding='SAME', strides=1, activation=tf.nn.relu)
# Pooling Layer 1
PL2 = tf.layers.max_pooling2d(inputs=CL2, pool_size=[2,2], padding='SAME', strides=2)
# Fully Connected (FC) Layer
L_flat = tf.reshape(PL2, [-1,7*7*64])
W3 = tf.Variable(tf.random_normal([7*7*64,10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10]))

# Model, Cost, Train
model_LC = tf.matmul(L_flat, W3) + b3
model = tf.nn.softmax(model_LC)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_LC, labels=Y))
train = tf.train.AdamOptimizer(0.01).minimize(cost)

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)), tf.float32))

# Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Training
	t1 = time.time()
	for epoch in range(training_epochs):
		total_batch = int(mnist.train.num_examples / batch_size)
		for i in range(total_batch):
			train_images, train_labels = mnist.train.next_batch(batch_size)
			c, _ = sess.run([cost, train], feed_dict={X: train_images, Y: train_labels})
			if i % 10 == 0:
				print('epoch:', epoch, ', batch number:', i) 
	t2 = time.time()
	# Testing
	print('Training Time (Seconds): ', t2-t1)
	print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

