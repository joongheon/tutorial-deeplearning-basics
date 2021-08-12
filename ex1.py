import tensorflow as tf
node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)
tf.print("node1 + node2 = ",node3)


