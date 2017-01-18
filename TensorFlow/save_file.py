import tensorflow as tf
import numpy as np

## 保存到文件

# W = tf.Variable([1,2,3], [3,4,5], dtype=tf.float32, name="weights")
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name="biases")
# init = tf.global_variables_initializer()

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "/logs/save_net.ckpt")
#     print("successfully saved the path: ", save_path)

# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable([1,2,3], [3,4,5], dtype=tf.float32, name="weights")
b = tf.Variable([[1,2,3]], dtype=tf.float32, name="biases")

#not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/logs/save_net.ckpt")
    print("Weights: ", sess.run(W))
    print("biases: ", sess.run(b))