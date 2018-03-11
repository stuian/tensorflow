# 训练直线y=0.1x+0.3

import tensorflow as tf 
import numpy as np 

# 数据
x = np.random.rand(1000)
# print(x)
bias = np.random.normal(loc=0,scale=1,size=None)
y = 0.1*x + 0.3 + bias
# print(y)

w = tf.Varible([1.0],dtype=tf.float32)
b = tf.Varible([0.0],dtype=tf.float32)


