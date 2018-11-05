"""
@author: Daniel
@contact: 
@file: pretrain.py
@time: 2018/11/2 14:17
"""

import tensorflow as tf
import numpy as np
import cv2
from pprint import pprint

import random
print("pretrain.py")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train_datas=[]
train_labels=[]

test_datas=[]
test_labels=[]

train_percentages=50
test_percentages=10

for k in range(1,6):
    file=r'/Users/daniel/FaceDetect/NetRunning/cifar-10-batches-py/data_batch_{}'.format(k)
    dic=unpickle(file)
    # # print(dic)
    # k=5
    #
    # print(dic[b'labels'][k])
    # m=(dic[b'data'][k])
    # l=(np.reshape(m,[32,32,3]))
    # print(l.shape)
    # pprint(l)
    # cv2.imshow('img',l)
    # cv2.waitKey(0)
    # print(dic[b'filenames'][k])

    nums=len(dic[b'labels'])
    for i in range(nums):
        label=dic[b'labels'][i]
        data=np.reshape(dic[b'data'][i],[3,32,32])

        # q=[[[]]]
        # q[:,:,0]=data[0,:,:]
        # q[:,:,1]=data[1,:,:]
        # q[:,:,2]=data[2,:,:]
        data=np.transpose(data, (2, 1, 0))
        # cv2.imshow('img',q)
        # cv2.waitKey(0)
        data=(data-128)/256
        chance=random.randint(0,train_percentages+test_percentages)
        if chance<train_percentages+test_percentages:
            train_datas.append(data)
            train_labels.append(label)
        else:
            test_datas.append(data)
            test_labels.append(label)
state=np.random.get_state()
np.random.shuffle(train_datas)
np.random.set_state(state)
np.random.shuffle(train_labels)
print("训练集打乱！")

np.save('data{}.npy'.format(k),[train_datas,train_labels,test_datas,test_labels])
print("保存文件结束！")