# MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data_MNIST", one_hot = True)

# Setting
import tensorflow as tf
import time
num_steps = 5000
batch_size =128
nH1 = 256
nH2 = 256
nH3 = 256

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

def mlp_LC(img):
	HL1 = tf.nn.relu(tf.add(tf.matmul(img, W['HL1']), b['HL1']))
	HL2 = tf.nn.relu(tf.add(tf.matmul(HL1, W['HL2']), b['HL2']))
	HL3 = tf.nn.relu(tf.add(tf.matmul(HL2, W['HL3']), b['HL3']))
	Out = tf.matmul(HL3, W['Out']) + b['Out']
	return Out
W = {
	'HL1' : tf.Variable(tf.random_normal([784, nH1])),
	'HL2' : tf.Variable(tf.random_normal([nH1, nH2])),
	'HL3' : tf.Variable(tf.random_normal([nH2, nH3])),
	'Out' : tf.Variable(tf.random_normal([nH3, 10]))
}
b = {
	'HL1' : tf.Variable(tf.random_normal([nH1])),
	'HL2' : tf.Variable(tf.random_normal([nH2])),
	'HL3' : tf.Variable(tf.random_normal([nH3])),
	'Out' : tf.Variable(tf.random_normal([10]))
}

# Model, Cost, Train
model_LC = mlp_LC(X)
model = tf.nn.softmax(model_LC)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_LC, labels=Y))
train = tf.train.AdamOptimizer(0.01).minimize(cost)

# Accuracy Computation
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model, 1), tf.argmax(Y,1)), tf.float32))

# Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Train
	t1 = time.time()
	for step in range(1, num_steps+1):
		train_images, train_labels = mnist.train.next_batch(batch_size)
		sess.run(train, feed_dict={X: train_images, Y: train_labels})
		if step % 500 == 0:
			print(step)
	t2 = time.time()
	print("Training Time (seconds): ", t2-t1)
	print("Testing Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

