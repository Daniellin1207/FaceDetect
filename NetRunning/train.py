# _*_ coding: utf-8 _*_
# @Time : 2018/11/4 上午11:00
# @Author :Daniel
# @File : train.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import NetRunning.NetDefine as Net
import time
import os

cls=10
lr=1e-3
Input_data='data5.npy'
batch_size=32
model_path = 'model/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
def train(lr):
    datas=np.load(Input_data)
    train_datas=datas[0]
    train_labels=datas[1]
    test_datas=datas[2]
    test_labels=datas[3]

    # print(test_labels)
    # print(test_datas[1].shape)
    # iters=len(train_labels)//batch_size+1

    x=tf.placeholder(dtype=tf.float32,shape=[None,32,32,3],name='input')
    y=tf.placeholder(dtype=tf.int64,shape=[None],name='label')

    keep_prob=tf.placeholder(dtype=tf.float32)
    output=Net.Net(x,keep_prob,cls)

    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=y)
    loss=tf.reduce_mean(loss)
    train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    temp=tf.cast(tf.equal(y,tf.argmax(output,axis=1)),tf.float32)
    accu=tf.reduce_mean(temp)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        # saver.restore(sess, '/Users/daniel/FaceDetect/NetRunning/model/model')

        start,end=0,batch_size
        i=1
        # for i in range(1,iters+1):
        while True:
            i+=1
            _,loss_value=sess.run([train_step,loss],feed_dict={x:train_datas[start:end],y:train_labels[start:end],keep_prob:0.5})
            if i %10 == 0:
                print("After {} steps, loss = {}, time = {}".format(i, loss_value, time.ctime()))
            if i %100==0:
                accuracy=sess.run(accu,
                                  # feed_dict={x: train_datas[start:end], y: train_labels[start:end], keep_prob: 1}
                                  feed_dict={x:test_datas,y:test_labels,keep_prob:1 }
                                  )
                # print(output)
                print("\nAfter {} steps, accuracy = {}, time = {}\n".format(i,accuracy,time.ctime()))
                saver.save(sess, model_path)
            start = end
            if end==len(train_labels):
                start,end=0,batch_size
            else:
                end=min(end+batch_size,len(train_labels))
        # accuracy = sess.run(accu, feed_dict={x: test_datas, y: test_labels, keep_prob: 1})
        # print("\nAfter {} steps, accuracy = {}, time = {}\n".format(i, accuracy, time.ctime()))
if __name__ == '__main__':
    # for lr in [
    #     1e-4,
    #     1e-5,
    #     # 1e-6,
    #     # 1e-7,
    #     1e-8,
    #     1e-9]:
    #     train(lr)
    train(lr)
# 0.09780000150203705
# 0.09640000015497208