# MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data_MNIST", one_hot=True)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Training Params
num_steps = 1
batch_size = 128

# Network Params
dim_image = 784 # 28*28 pixels
nHL_thief = 256
nHL_police = 256
dim_noise = 100 # Noise data points

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
	return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

W = {
	'HL_thief' : tf.Variable(glorot_init([dim_noise, nHL_thief])),
	'OL_thief' : tf.Variable(glorot_init([nHL_thief, dim_image])),
	'HL_police': tf.Variable(glorot_init([dim_image, nHL_police])),
	'OL_police': tf.Variable(glorot_init([nHL_police, 1])),
}
b = {
	'HL_thief' : tf.Variable(tf.zeros([nHL_thief])),
	'OL_thief' : tf.Variable(tf.zeros([dim_image])),
	'HL_police': tf.Variable(tf.zeros([nHL_police])),
	'OL_police': tf.Variable(tf.zeros([1])),
}

# Neural Network: Thief
def nn_thief(x):
	HL = tf.nn.relu(tf.add(tf.matmul(x, W['HL_thief']), b['HL_thief']))
	OL = tf.nn.sigmoid(tf.add(tf.matmul(HL, W['OL_thief']), b['OL_thief']))
	return OL

# Neural Network: Police
def nn_police(x):
	HL = tf.nn.relu(tf.add(tf.matmul(x, W['HL_police']), b['HL_police']))
	OL = tf.nn.sigmoid(tf.add(tf.matmul(HL, W['OL_police']), b['OL_police']))
	return OL

# Network Inputs
IN_THIEF  = tf.placeholder(tf.float32, shape=[None, dim_noise])
IN_POLICE = tf.placeholder(tf.float32, shape=[None, dim_image])

# Build Thief/Generator Neural Network
sample_thief = nn_thief(IN_THIEF)

# Build Police/Discriminator Neural Network (one from noise input, one from generated samples)
police_data_real = nn_police(IN_POLICE)
police_data_fake = nn_police(sample_thief)
vars_thief  = [W['HL_thief'],  W['OL_thief'],  b['HL_thief'],  b['OL_thief'] ]
vars_police = [W['HL_police'], W['OL_police'], b['HL_police'], b['OL_police']]

# Cost, Train
cost_thief   = -tf.reduce_mean(tf.log(police_data_fake))
cost_police  = -tf.reduce_mean(tf.log(police_data_real) + tf.log(1. - police_data_fake))
train_thief  = tf.train.AdamOptimizer(0.0002).minimize(cost_thief,  var_list=vars_thief)
train_police = tf.train.AdamOptimizer(0.0002).minimize(cost_police, var_list=vars_police)

# Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1, num_steps+1):
		# Prepare Data
		# Get the next batch of MNIST data (only images are needed, not labels)
		batch_images, _ = mnist.train.next_batch(batch_size)
		# Generate noise to feed to the generator/thief
		z = np.random.uniform(-1., 1., size=[batch_size, dim_noise])
		# Train
		sess.run([train_thief, train_police], feed_dict = {IN_POLICE: batch_images, IN_THIEF: z})
		# Generate images from noise, using the generator network.
		f, a = plt.subplots(4, 10, figsize=(10, 4))
		for i in range(10):
			# Noise input
			z = np.random.uniform(-1., 1., size=[4, dim_noise])
			g = sess.run([sample_thief], feed_dict={IN_THIEF: z})
			g = np.reshape(g, newshape=(4, 28, 28, 1))
			# Reverse colors for better display
			g = -1 * (g - 1)
			for j in range(4):
				# Generate image from noise. Extend to 3 channels for matplot figure.
				img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
				a[j][i].imshow(img)
		print(i)
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
