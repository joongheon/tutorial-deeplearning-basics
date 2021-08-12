####################################
import tensorflow as tf
node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)
tf.print("node1 + node2 = ",node3)
######################################
import tensorflow as tf

a = tf.constant(0, shape=[3, 1])
b = tf.constant(0, shape=[1, 3])
@tf.function
def forward(x,y):
  return tf.matmul(x,y) 

out_c = forward([[3],[2],[1]],[[1,2,3]])
print(out_c)
######################################
import tensorflow as tf
import numpy as np

x_data = [[1,1],[2,2],[3,3]]
y_data = [[10],[20],[30]]
train_step = 2001

W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

@tf.function
def linear_regression(x):
  return W*x+b

@tf.function
def mean_square(y_p,y_t):
  return tf.reduce_mean(tf.square(y_p-y_t))

optimizer = tf.optimizers.SGD(0.01)

def run_optimization():
  with tf.GradientTape() as g:
    pred = linear_regression(x_data)
    loss = mean_square(pred,y_data)
  gradients = g.gradient(loss,[W,b])
  optimizer.apply_gradients(zip(gradients,[W,b]))
for step in range(1,train_step):
  run_optimization() 
  if step % 100 == 0:
    pred = linear_regression(x_data)
    loss = mean_square(pred, y_data)
    print("step : %i, loss : %f, W : %f, b: %f" %(step,loss,W.numpy(),b.numpy())) 

import matplotlib.pyplot as plt
plt.plot(x_data,y_data,'ro',label='real')
plt.plot(x_data,np.array(W*x_data+b),label='pred')
plt.legend()
plt.show() 
