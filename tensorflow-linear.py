# 训练直线y=0.1x+0.3

import tensorflow as tf 
import numpy as np 

# 数据
x_data = np.random.rand(1000)
# print(x)
for i in x_data:
	bias = np.random.normal(loc=0,scale=1,size=None)
	print(bias)
	y_data = 0.1*i + 0.3 + bias
# print(y)

learning_rate = 0.001

w = tf.Variable([0.2],dtype=tf.float32)
b = tf.Variable([0.1],dtype=tf.float32)

x_ = tf.placeholder(tf.float32,(None))
y_ = tf.placeholder(tf.float32,(None))

y = x_*w + b

loss = tf.reduce_mean(tf.square(y-y_))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
  		curr_train,curr_W, curr_b, curr_loss = sess.run([train,w, b, loss], {x_: x_data, y_: y_data})

  		if i % 20 == 0:
		    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))