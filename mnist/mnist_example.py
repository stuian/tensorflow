import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#两个张量
X = tf.placeholder(tf.float32,shape=(None,784))
y_ = tf.placeholder(tf.float32,shape=(None,10))

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(X,W)+b)

#所有图片的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#学习率为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#important step
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#为什么是1，预测的概率不一定最大是1，所有概率加起来是1吧
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

for i in range(1000):
    #该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict = {X:batch_xs,y_:batch_ys})
    if i % 50 == 0:
        print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))