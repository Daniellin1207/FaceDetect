"""
@author: Daniel
@contact: 
@file: preprocess.py
@time: 2018/10/22 9:43
"""

import os
import json
import glob
import numpy as np
import cv2

print("preprocess.py")



def testZoom():
    file=r"C:\Users\lyl8373\Desktop\single\333\92.jpg"
    cont=open(file.replace('.jpg','.txt'),'r',encoding='utf-8').read()
    for cont in cont.split('\n'):
        jd=json.loads(cont)

        x1, y1, x2, y2=map(int,jd["rect"].split(','))
        print(x1, y1, x2, y2)
        img=cv2.imread(file)

        # img1=cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))
        # cv2.imshow("img",img1)
        # cv2.waitKey(0)

        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
        img=cv2.rectangle(img,(x1*400//1280,y1*400//720),(x2*400//1280,y2*400//720),(255,0,0),thickness=1)
        cv2.imshow("img",img)
        cv2.waitKey(0)

# testZoom()

def preProcess():
    outDir = "dataset"
    outFile = "dataset.npy"
    inDir = r'C:\Users\lyl8373\Desktop\single'
    inFile = ""

    train_percentage = 5
    validation_percentage = 2
    test_percentage = 3

    training_images=[]
    training_labels=[]

    testing_images=[]
    testing_labels=[]

    validation_images=[]
    validation_labels=[]

    widthOld=1280
    heightOld=720

    widthNew=224
    heightNew=224

    ps=glob.glob(inDir+'/*/*.jpg')
    for file in ps:
        img = cv2.imread(file)
        img = cv2.resize(img, (widthNew, heightNew))
        try:
            cont=open(file.replace('.jpg','.txt')).read()
            if cont.count("{")>1:
                print(file,"跳过！")
                continue
            x1,y1,x2,y2=map(int,json.loads(cont)["rect"].split(","))
            label=[(x1+x2)/2*widthNew//widthOld,(y1+y2)//2,(x2-x1)*widthNew//heightOld,(y2-y1)*heightNew//heightOld]

            chance = np.random.randint(train_percentage+validation_percentage+test_percentage)
            if chance<train_percentage:
                training_images.append(img)
                training_labels.append(label)
            elif chance<train_percentage+validation_percentage:
                validation_images.append(img)
                validation_labels.append(label)
            else:
                testing_images.append(img)
                testing_labels.append(label)
            print(file,"处理完成！")
        except:
            print(file)
            input()

    state=np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    print("random打乱！")

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    np.save(os.path.join(outDir,outFile),np.asarray([training_images,
                                                           training_labels,
                                                           validation_images,
                                                           validation_labels,
                                                           testing_images,
                                                           testing_labels]))
preProcess()