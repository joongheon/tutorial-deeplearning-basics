import tensorflow as tf
a = tf.constant(0, shape=[3, 1])
b = tf.constant(0, shape=[1, 3])
def forward(x,y):
  return tf.matmul(x,y) 
out_c = forward([[3],[2],[1]],[[1,2,3]])
print(out_c)