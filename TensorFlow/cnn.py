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
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result;

def weight_variable(shape):
    initial =tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#定义输入网络中的占位符
xs = tf.placeholder(tf.float32, [None, 784]) #images
ys = tf.placeholder(tf.float32, [None, 10]) #labels
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
print(x_image.shape)

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32])                       #patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    #输出尺寸为:28*28*32
h_pool1 = max_pool_2x2(h_conv1)                             #输出尺寸为:14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64])                       #patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv2)    #输出尺寸为:14*14*64
h_pool2 = max_pool_2x2(h_conv2)                             #输出尺寸为:7*7*64

## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
#[n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_))

#添加输出层
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) #误差

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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