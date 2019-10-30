# MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data_MNIST", one_hot=True)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Training Params
num_steps = 100000
batch_size = 128

# Network Params
dim_image = 784 # 28*28 pixels
nHL_G = 256
nHL_D = 256
dim_noise = 100 # Noise data points

# A custom initialization (Xavier Glorot init)
def glorot_init(shape):
	return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

W = {
	'HL_G' : tf.Variable(glorot_init([dim_noise, nHL_G])),
	'OL_G' : tf.Variable(glorot_init([nHL_G, dim_image])),
	'HL_D' : tf.Variable(glorot_init([dim_image, nHL_D])),
	'OL_D' : tf.Variable(glorot_init([nHL_D, 1])),
}
b = {
	'HL_G' : tf.Variable(tf.zeros([nHL_G])),
	'OL_G' : tf.Variable(tf.zeros([dim_image])),
	'HL_D' : tf.Variable(tf.zeros([nHL_D])),
	'OL_D' : tf.Variable(tf.zeros([1])),
}

# Neural Network: Generator
def nn_G(x):
	HL = tf.nn.relu(tf.add(tf.matmul(x, W['HL_G']), b['HL_G']))
	OL = tf.nn.sigmoid(tf.add(tf.matmul(HL, W['OL_G']), b['OL_G']))
	return OL

# Neural Network: Discriminator
def nn_D(x):
	HL = tf.nn.relu(tf.add(tf.matmul(x, W['HL_D']), b['HL_D']))
	OL = tf.nn.sigmoid(tf.add(tf.matmul(HL, W['OL_D']), b['OL_D']))
	return OL

# Network Inputs
IN_G = tf.placeholder(tf.float32, shape=[None, dim_noise])
IN_D = tf.placeholder(tf.float32, shape=[None, dim_image])

# Build Thief/Generator Neural Network
sample_G = nn_G(IN_G)

# Build Police/Discriminator Neural Network (one from noise input, one from generated samples)
D_real = nn_D(IN_D)
D_fake = nn_D(sample_G)
vars_G = [W['HL_G'], W['OL_G'], b['HL_G'], b['OL_G']]
vars_D = [W['HL_D'], W['OL_D'], b['HL_D'], b['OL_D']]

# Cost, Train
cost_G = -tf.reduce_mean(tf.log(D_fake))
cost_D = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
train_G = tf.train.AdamOptimizer(0.0002).minimize(cost_G, var_list=vars_G)
train_D = tf.train.AdamOptimizer(0.0002).minimize(cost_D, var_list=vars_D)

# Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1, num_steps+1):
		# Get the next batch of MNIST data
		batch_images, _ = mnist.train.next_batch(batch_size)
		# Generate noise to feed to the generator G
		z = np.random.uniform(-1., 1., size=[batch_size, dim_noise])
		# Train
		sess.run([train_G, train_D], feed_dict = {IN_D: batch_images, IN_G: z})
		f, a = plt.subplots(4, 10, figsize=(10, 4))
		for i in range(10):
			z = np.random.uniform(-1., 1., size=[4, dim_noise])
			g = sess.run([sample_G], feed_dict={IN_G: z})
			g = np.reshape(g, newshape=(4, 28, 28, 1))
			# Reverse colors for better display
			g = -1 * (g - 1)
			for j in range(4):
				# Generate image from noise. Extend to 3 channels for matplot figure.
				img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
				a[j][i].imshow(img)
	f.show()
	plt.draw()
	plt.waitforbuttonpress()

'''
# Update
sudo apt-get update

# TensorFlow
sudo apt-get install python-pip
pip install tensorflow

# matplotlib
pip install matplotlib
sudo apt-get install python-tk
'''
