import tensorflow as tf
import numpy as np
x_data = [[1,1],[2,2],[3,3]]
y_data = [[10],[20],[30]]
W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

def linear_regression(x):
  return W*x+b
def mean_square(y_p,y_t):
  return tf.reduce_mean(tf.square(y_p-y_t))
def run_optimization():
  with tf.GradientTape() as g:
    model = linear_regression(x_data)
    cost = mean_square(model,y_data)
  gradients = g.gradient(cost,[W,b])
  tf.optimizers.SGD(0.01).apply_gradients(zip(gradients,[W,b]))

for step in range(1,2001):
  run_optimization() 
  if step % 100 == 0:
    model = linear_regression(x_data)
    cost = mean_square(model, y_data)
    print("step : %i, cost : %f, W : %f, b: %f" %(step,cost,W.numpy(),b.numpy())) 



    