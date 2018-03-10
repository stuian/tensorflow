
# coding: utf-8

# In[7]:

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

#%% Reading data

def read_cifar10(data_dir, is_train, batch_size, shuffle):
    """Read CIFAR10
    
    Args:
        data_dir: the directory of CIFAR10
        is_train: true：训练，flase：测试
        batch_size:
        shuffle:       
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    
    """
    img_width = 32
    img_height = 32
    img_depth = 3
    
    label_bytes = 1        # 1+3*32*32 = 3073
    image_bytes = img_width*img_height*img_depth
    
    
    with tf.name_scope('input'):
        
        # train or test
        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %ii)
                                        for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            
        #build a queue
        filename_queue = tf.train.string_input_producer(filenames)
        
        #读取队列 每一次读取3073个byte
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)
        
        
        key, value = reader.read(filename_queue)
           
        record_bytes = tf.decode_raw(value, tf.uint8)
        
        #切分，从0-1 分给label
        label = tf.slice(record_bytes, [0], [label_bytes])   
        label = tf.cast(label, tf.int32)
        
        #切分，从1-3073给image_raw
        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])     
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])     
        image = tf.transpose(image_raw, (1,2,0)) # convert from D/H/W to H/W/D       
        image = tf.cast(image, tf.float32)

     
        # data argumentation

#        image = tf.random_crop(image, [24, 24, 3])# randomly crop the image size to 24 x 24
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_brightness(image, max_delta=63)
#        image = tf.image.random_contrast(image,lower=0.2,upper=1.8)


        #图片做标准化
#         image = tf.image.per_image_standardization(image) #substract off the mean and divide by the variance 

        #打乱顺序
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = batch_size,
                                    num_threads= 16,
                                    capacity = 2000,
                                    min_after_dequeue = 1500)
        else:
            images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 16,
                                    capacity= 2000)


        ## ONE-HOT
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])

        return images, label_batch
        
    #return images, tf.reshape(label_batch, [batch_size])

    





# In[10]:

# # To test the generated batches of images
# # When training the model, DO comment the following codes

# data_dir = '../../data/cifar-10-batches-bin/'
# BATCH_SIZE = 4
# image_batch, label_batch = read_cifar10(data_dir,
#                                        is_train=True,
#                                        batch_size=BATCH_SIZE, 
#                                        shuffle=True)

# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
   
#    try:
#        while not coord.should_stop() and i<1:
           
#            img, label = sess.run([image_batch, label_batch])
# #            print(label[j])
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
           
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


# In[ ]:



