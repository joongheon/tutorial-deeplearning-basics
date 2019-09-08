from keras.utils import np_utils
from keras.datasets import cifar10
import tensorflow as tf
import time

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_labels = np_utils.to_categorical(train_labels) # One-Hot Encoding
test_labels = np_utils.to_categorical(test_labels) # One-Hot Encoding

training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 32,32,3])
Y = tf.placeholder(tf.float32, [None, 10])
X_img = tf.reshape(X, [-1, 32, 32, 3])
  
# Convolution Layer 1
W1 = tf.Variable(tf.random_normal([3,3,3,32], stddev=0.01))
CL1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
CL1 = tf.nn.relu(CL1)
# Pooling Layer 1
PL1 = tf.nn.max_pool(CL1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Convolution Layer 2
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
CL2 = tf.nn.conv2d(PL1, W2, strides=[1,1,1,1], padding='SAME')
CL2 = tf.nn.relu(CL2)
# Pooling Layer 2
PL2 = tf.nn.max_pool(CL2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Fully Connected (FC) Layer
L_flat = tf.reshape(PL2, [-1,8*8*64])
W3 = tf.Variable(tf.random_normal([8*8*64,10], stddev=0.01))
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
		total_batch = int(train_images.shape[0] / batch_size)
		for i in range(total_batch):
			batch_train_images = train_images[i*batch_size:(i+1)*batch_size]
			batch_train_labels = train_labels[i*batch_size:(i+1)*batch_size]
			c, _ = sess.run([cost, train], feed_dict={X: batch_train_images, Y: batch_train_labels})
			if i % 10 == 0:
				print('epoch:', epoch, ', batch number:', i) 
	t2 = time.time()
	# Testing
	print('Training Time (Seconds): ', t2-t1)
	total_batch = int(test_images.shape[0] / batch_size)
	for i in range(total_batch):
		batch_test_images = test_images[i*batch_size:(i+1)*batch_size]
		batch_test_labels = test_labels[i*batch_size:(i+1)*batch_size]
	print('Accuracy: ', sess.run(accuracy, feed_dict={X: batch_test_images, Y: batch_test_labels}))

