import tensorflow as tf
import numpy as np
sample = " My name is Joongheon Kim."
idx2char = list(set(sample))  # index -> char
print(idx2char)
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex
print(char2idx)
# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
print(dic_size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of RNN rollings (unit #)
print(sequence_length)
sample_idx = [char2idx[c] for c in sample]  # char to index
print(sample_idx)
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1)
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n)
X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label
x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 ... 0
# cell and RNN
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32)
# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=tf.ones([batch_size, sequence_length]))
cost = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(0.1).minimize(cost)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        l, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, ", loss:", l, ", Prediction:", ''.join(result_str))
		
		