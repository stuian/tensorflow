pic_height = 28
pic_width = 28
pic_size = pic_height*pic_width
pic_class= 10
n_channel = 1

n_inputs = pic_size
n_outputs = pic_class

n_filters_convl = 32
n_filters_conv2 = 64
n_inputs_full1 = 7*7*n_filters_conv2
n_neurons_fulll = 96
n_inputs_full2 = n_W_full2
n_neurons_full2 = n_outputs

filter_height = 5
filter_width = 5

batch_size = 100
n_train_batches = int(np.ceil(mnist.train.num_examples))
n_test_batches = int(np.ceil(mnist.test.num_examples))

n_epochs = 1

#函数定义
#定义权值函数
def weight(shape):
	init = tf.truncated_normal(shape，stddev = 2)
	return tf.Variable(init)

#定义偏置值
def bias(shape):
	init = tf.constant(.0, shape = shape)
	return tf.Variable(init)

#定义卷积操作
def conv2(x, W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

#池化操作
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')

#定义placeholder
x = tf.placeholder(tf.float32,[None,n_inputs])
y = tf.placeholder(tf.float32,[None,n_outputs])

#定义神经网络模型

#把输入的图像格式转换成第一层卷积网络输入所要求的格式
x_images = tf.reshape(x,[-1,pic_height,pic_width,n_channel])
#第一层卷积网络的权值和偏置值
W_conv1 = weight([filter_height,filter_width,n_channel,n_filters_convl])
b_conv1 = bias ([n_filters_convl])

#卷积
sigma_conv1 = tf.nn.relu(sigma_conv1)
#池化
pool_conv1 = max_pool_2x2(relu_conv1)

#第二层卷积网络的权值和偏置值
W_conv2 = weight([filter_height,filter_width,n_filters_convl,n_filters_conv2])
b_conv2 = bias([n_filters_conv2])

#卷积
sigma_conv2 = conv2d((pool_conv1,W_conv2)+b_conv2)
#非线性变换
relu_conv2 = tf.nn.relu(sigma_conv2)
#池化
pool_conv2 = max_pool_2x2(relu_conv2)

#把卷积层输出的格式转换成全连接层的输入的格式
pool_flat_conv2 = tf.reshape(pool_conv2,[-1,n_inputs_full1])

#第一层全连接层
W_full1 = weight([n_inputs_full1,n_neurons_fulll])
b_full1 = bias([n_neurons_fulll])
#加法
sigma_full1 = tf.matmul(pool_flat_conv1,W_full1) + b_full1
#非线性变换
tanh_full1 = tf.nn.tanh(sigma_full1) + b_conv2

#第二层全连接层
W_full2 = weight([n_neurons_fulll,n_neurons_full2])
b_full2= bias(n_neurons_full2)

#加法
sigma_full2 = tf.matmul(tanh_full1,W_full2) + b_full2
#非线性变换
softmax_full2= tf.nn.softmax(sigma_full2)
prediction = softmax_full2

#损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = )
loss = tf.reduce_mean(cross_entropy)

#最优化算法
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize（loss）


#定义测试指标
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#建立会话，进行训练和测试

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		for batch in range(n_train_batches):
			x_batch,y_batch = mnist.train.next_batch(batch_size)
			_,acc = sess.run([train,accuracy],feed_dict = {x:x_batch,y:y_batch})
			train_acc += acc
		train_acc /= n_train_batches

		for batch in range(n_test_batches):
			x_batch,y_batch =  mnist.test.next_batch(batch_size)
			acc = sess.run(accuracy,feed_dict = {x:x_batch,y:y_batch})
			test.acc += acc
		tets_acc /= n_test_batches

		print("Epoch" + str(epoch) + ":Test_acc: " + str(Test_acc) + 
			":train_acc:" + str(strain_acc) + ":")
