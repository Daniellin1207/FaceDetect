"""
@author: Daniel
@contact: 
@file: test.py
@time: 2018/11/2 13:39
"""

import os
import shutil
import traceback
import json
import time
import glob

print("test.py")

def f(a,b,*c,d):
    for i in c:
        print(i)
    assert len([*c])==2
    l=len([*c])
    print(l)

    print([a,b,*c,d])

# f(10,20,3,32,2,d=6)
# n=int(input())
# x=0
# while True:
#   if n==1:
#     break
#   if n%2==0:
#      n=n//2
#   else:
#     n=(3*n+1)//2
#   x+=1
# print(x)

# x = input()
# dic={'1':'yi','2':'er','3':'san','4':'si','5':'wu','6':'liu','7':'qi','8':'ba','9':'jiu','0':'ling'}
# n=str(sum(map(int,x)))
# out=[]
# for w in n:
#     out.append(dic[w])
# print(' '.join(out))

l=[0,1,0,0,0]
import tensorflow as tf
m=tf.argmax(l)
sess=tf.Session()
print(sess.run(m))
# print(m.eval())
# print(m.eval)