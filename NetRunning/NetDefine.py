"""
@author: Daniel
@contact: 
@file: NetDefine.py
@time: 2018/10/22 10:35
"""

import os
import tensorflow as tf
import numpy as np
print("NetDefine.py")

# data_dict=np.load('vgg16.npy',encoding='latin1').item()

def print_layer(t):
    print(t.op.name,t.get_shape().as_list())

def max_pool(x,name):
    activation=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
    print_layer(activation)
    return activation

def conv(x,d_out,scope_name,
         # *kernel1,
         finetune=False):
    d_in=x.get_shape()[-1].value
    # assert len([*kernel1])==2
    with tf.variable_scope(scope_name):
        if finetune:
            kernel=tf.get_variable('weights',initializer=tf.constant(data_dict[scope_name][0]))
            bias=tf.get_variable('bias',initializer=tf.constant(data_dict[scope_name][1]))
            print('fineturn')
        else:
            kernel=tf.Variable(tf.truncated_normal([3,3,
                # *kernel1,
                d_in,d_out],stddev=0.1))
            bias=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[d_out]),trainable=True,name='bias')
            print("truncated_normal")
        conv=tf.nn.conv2d(x,kernel,[1,1,1,1],padding='SAME')
        activation=tf.nn.relu(conv+bias)
        print_layer(activation)
        return activation

def fc(x,d_out,scope_name,fineturn=False):
    d_in=x.get_shape()[-1].value
    with tf.variable_scope(scope_name):
        if fineturn:
            weights=tf.get_variable('weights',initializer=tf.constant(data_dict[scope_name][0]))
            bias=tf.get_variable('bias',initializer=tf.constant(data_dict[scope_name][1]))
            print('fc finetune')
        else:
            weights=tf.Variable(tf.truncated_normal([d_in,d_out],stddev=0.1))
            bias=tf.Variable(tf.truncated_normal([d_out],stddev=0.1),trainable=True,name='bias')
            print("fc truncated_normal")
        activation=tf.add(tf.matmul(x,weights),bias)
        print_layer(activation)
        return activation

def Net(img,_dropout,cls):
    # img=32*32*3
    conv1_1=conv(img,6,"conv1_1",finetune=False)
    conv1_2=conv(conv1_1,6,"conv1_2",finetune=False)
    max_pool1_3=max_pool(conv1_2,'max_pool1_3')
    # 16*16*6
    conv2_1=conv(max_pool1_3,12,"conv2_1",finetune=False)
    conv2_2=conv(conv2_1,12,"conv2_2",finetune=False)
    max_pool2_3=max_pool(conv2_2,"max_pool2_3")
    # 8*8*12
    conv3_1=conv(max_pool2_3,24,"conv3_1",finetune=False)
    conv3_2=conv(conv3_1,24,"conv3_2",finetune=False)
    max_pool3_3=max_pool(conv3_2,"max_pool3_3")
    # 4*4*24
    flatten=tf.reshape(max_pool3_3,[-1,4*4*24])
    fc4=fc(flatten,60,'fc4',fineturn=False)
    dropout1 = tf.nn.dropout(fc4, _dropout)
    fc5=fc(dropout1,cls,'fc5',fineturn=False)
    return fc5
