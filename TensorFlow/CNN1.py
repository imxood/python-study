import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides [1, x_movement, y_movement, 1]
    # 必须有: strides[0]=strides[3]=1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    # strides [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# 定义输入网络的占位符
xs = tf.placeholder(tf.float32, [None, 784]) #28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape) # [n_samples, 28, 28, 1]

## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

#[n_samples, 7, 7, 64] ->>[n_samples, 7*7*64]


## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch 5*5, int size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) +b_conv1) # out size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                         # out size 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5*5, int size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(x_image, W_conv1) +b_conv1) # out size 14*14*64
h_pool2 = max_pool_2x2(h_conv1)                         # out size 7*7*64