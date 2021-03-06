# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import tensorflow.python.debug as tf_debug

from tensorflow.examples.tutorials.mnist import input_data

#加载数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#添加层,返回层的输出
def add_layer(inputs, in_size, out_size, activation_function=None):
    
    #权重
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W") # 784*10矩阵

    #偏差
    biases = tf.Variable(tf.zeros([1, out_size])+0.1, name="b")

    #预测的还未激活的值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases # inputs: [None, 784]

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b) #激活

    return outputs

#预测值
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result;

#定义输入网络中的占位符
xs = tf.placeholder(tf.float32, [None, 784], name="x_input") #images
ys = tf.placeholder(tf.float32, [None, 10], name="y_input") #labels

#添加输出层
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) #误差

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys=mnist.train.next_batch(100)
        print("batch_xs.len: ", len(batch_xs), "batch_ys.len: ", len(batch_ys))
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i%50==0:
            pre = sess.run(prediction, feed_dict={xs: batch_xs, ys: batch_ys})
            print( type(pre), pre[0] )
            print( compute_accuracy(mnist.test.images, mnist.test.labels) )