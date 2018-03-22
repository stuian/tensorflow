
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np

#作用：1）make this notebook's output stable across runs
#2)会清空default_graph
def reset_graph(seed=318):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[3]:

from tensorflow.examples.tutorials.mnist import input_data


# In[4]:

data_path="/data/stu17/mnist"
data_path2="/data/stu17/mnist2"
#mnist=input_data.read_data_sets(data_path,one_hot=True)#one hot编码,
mnist=input_data.read_data_sets(data_path2)#raw coding,是一个数字 节省空间


# In[17]:

#图片的常量
pic_height=28
pic_width=28
pic_size=pic_height*pic_width
pic_classes=10
pic_channels=1
#网络拓扑的常量
n_inputs=pic_size
n_hidden1=50
n_hidden2=25
n_outputs=pic_classes

#迭代次数
n_epochs=1

#mini_batch
batch_size=10
n_train_batches=mnist.train.num_examples//batch_size
n_test_batches=mnist.test.num_examples//batch_size

#learning rate
learning_rate=1e-4


# # 构建模型

# In[18]:

reset_graph()


# In[19]:

# reset_graph()
# #定义网络的全联接层
# def neurons_layer(X,n_neurons,activation=None):
#     #获取输入单元个数
#     n_inputs=int(X.get_shape()[-1])#get_shape 返回的是tensor(batch_size,pic_size)
    
#     #定义W
#     stddev=2/np.sqrt(n_inputs)#经验值
#     init=tf.truncated_normal((n_inputs,n_neurons),mean=0,stddev=stddev)
#     W=tf.Variable(init)
    
#     #定义b
#     b=tf.Variable(tf.zeros((n_neurons),tf.float32))
    
#     #做加法
#     sigma=tf.matmul(X,W)+b
    
#     #非线性变换
#     if activation is not None:
#         return activation(sigma)
    
#     #不做非线性变换
#     return sigma


# ### 构建神经网络

# In[20]:

#定义placeholder,(batch_size'=='None)
X=tf.placeholder(tf.float32,(None,pic_size))
#Y=tf.placeholder(tf.float32,(None,pic_classes))
Y=tf.placeholder(tf.int64,(None))


# In[21]:

# hidden1=neurons_layer(X,n_hidden1,tf.nn.relu)#(batch_size,n_hidden1)
# hidden2=neurons_layer(hidden1,n_hidden2,tf.nn.relu)#(n_hidden1,n_hidden2)
# prediction=neurons_layer(hidden2,n_outputs)#(batch_size,n_outputs)
hidden1=tf.layers.dense(X,n_hidden1,tf.nn.relu)
hidden1_dropout=tf.nn.dropout(hidden1,0.8)
hidden2=tf.layers.dense(hidden1,n_hidden2,tf.nn.relu)
hidden2_dropout=tf.nn.dropout(hidden2,0.8)
prediction=tf.layers.dense(hidden2,n_outputs)


# ### 交叉熵损失

# In[22]:

#xentropy=tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction)
xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=prediction)
loss=tf.reduce_mean(xentropy)


# ### 定义学习算法

# In[23]:

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(loss)


# ### 准确率（测试、验证用，也可以在训练时）

# In[24]:

#correct=tf.equal(tf.argmax(Y,1),tf.argmax(prediction,1))#y.get_shape=(batch_size,pic_classes)
correct=tf.nn.in_top_k(prediction,Y,1)
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))


# # 创建会话，进行训练测试

# In[28]:

#保存模型
saver=tf.train.Saver()
model_path="/data/stu17/my_model/mlp.ckpt"

#模型可视化
loss_summary=tf.summary.scalar('loss',loss)
acc_summary=tf.summary.scalar('acc',accuracy)
logdir='./logdir'
filewritter=tf.summary.FileWriter(logdir,tf.get_default_graph())
merged=tf.summary.merge_all()


# In[29]:

init=tf.global_variables_initializer()


# In[31]:

with tf.Session() as sess:
    sess.run(init)
    #saver.restore(sess,model_path)
    
    #进行多趟的训练和测试
    for epoch in range(n_epochs):
        
        #mini_batch
        test_acc=.0
        train_acc=.0
        for batch in range(n_train_batches):
            x_batch,y_batch=mnist.train.next_batch(batch_size)
            #fetch,公共的部分放在list中算一次
            train_val,acc_val=sess.run([train,accuracy],
                               feed_dict={X:x_batch,Y:y_batch})
            train_acc+=acc_val
            
            merge_str=sess.run(merged,feed_dict={X:x_batch,Y:y_batch})
            filewritter.add_summary(merge_str,batch)
            
        train_acc/=n_train_batches
        
        for batch in range(n_test_batches):
            x_batch,y_batch=mnist.test.next_batch(batch_size)
            
            acc_bal=sess.run(accuracy,feed_dict={X:x_batch,Y:y_batch})
            test_acc+=acc_val
        test_acc/=n_test_batches
        
        print("Epoch"+str(epoch)+';'
             "Train_acc"+str(train_acc)+';'
             "Test_acc"+str(test_acc))
    saver.save(sess,model_path)


# In[ ]:




# In[ ]:




# In[ ]:



