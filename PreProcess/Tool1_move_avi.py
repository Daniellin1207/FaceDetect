"""
@author: Daniel
@contact: 
@file: Tool1_move_avi.py
@time: 2018/10/22 9:27
"""

import os
import shutil
import traceback
import json
import time
import glob

print("Tool1_move_avi.py")

path = input("请输入.avi文件所在路径: ")
savepath=input("请输入保存路径: ")
for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith('.avi'):

            if not os.path.exists(os.path.dirname(root.replace(path,savepath))):
                os.makedirs(os.path.dirname(root.replace(path,savepath)))

            os.rename(root,root.replace(path,savepath))
            print(root,"转移成功！")

            # os.rename(os.path.join(root,file).replace('.avi','.txt'),os.path.join(root,file).replace('.avi','.txt').replace(path,savepath))
            # print(os.path.join(root,file),"转移成功！")
