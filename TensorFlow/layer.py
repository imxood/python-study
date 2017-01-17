import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#添加层,返回层的输出
def add_layer(inputs, in_size, out_size, n_layer, activation_function):
    layer_name = 'layer%s' % n_layer
    #权重
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name+'/weights', Weights)

        with tf.name_scope("biases"):
            #偏差
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name="b")
            tf.summary.histogram(layer_name+'/biases', biases)

        #预测的值(还未激活)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram(layer_name+'/outputs', outputs)

        return outputs

#输入数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#模拟真实数据
noise = np.random.normal(0, 0.05, x_data.shape)
#输出数据
y_data = np.square(x_data) - 0.5 + noise


#定义输入网络中的占位符
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

#添加隐藏层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

#添加输入层
predition = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

#容错处理
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("/logs/", sess.graph)

sess.run(init)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i%50==0:
        #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # predition_value = sess.run(predition, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, predition_value, 'r-', lw=5)
       
        # plt.pause(0.1)
        result =  sess.run(merged, feed_dict={xs: x_data, ys: y_data})

        #每个50步一个点
        writer.add_summary(result, i)