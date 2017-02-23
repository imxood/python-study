import tensorflow as tf
import numpy as np

#创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#权重与偏差
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #Weights,大写,表示矩阵
biases = tf.Variable(tf.zeros([1]))

#预测
y = Weights * x_data + biases

#损失与优化训练
loss = tf.reduce_mean(tf.square(y-y_data)) #均方误差
optimizer = tf.train.GradientDescentOptimizer(0.5) #梯度下降算法,学习速率为0.5
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run(Weights), sess.run(biases))

#提升y的准确度




# print(x_data)