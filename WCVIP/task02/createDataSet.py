# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 12:42:54 2020
conda create -n EICT -y
conda activate EICT
conda install tensorflow -y
conda install -c conda-forge opencv -y
#pip install opencv-python
conda install -c conda-forge python-wget -y
conda install matplotlib -y
@author: ALI
"""

import os
import numpy as np
from random import shuffle
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2 
import _pickle as cPickle
import pickle

"""The dataset can be downloaded from 
https://drive.google.com/drive/folders/1ZiGxk6ZN5IBjtNAI4JkJK9YFehS8KNmT?usp=sharing

Let create Training and Testing Dataset
"""

rootDir='word_traintestset'



trainDir=rootDir+'/Train'
trainList=[]
count=0
for path, subdirs, files in os.walk(trainDir):
  for name in files:
    if not name.endswith('.png'):
      continue  
    count = count +1
    imPath=os.path.join(path, name); 
    image=cv2.imread(imPath,0) ### we want to read image as gray
    h,w=image.shape[:2]
    image=cv2.resize(image,(int(w*32/h),32))
    print('Train ',count,image.shape)
    parts=name.split('_');   
    trainList.append([image, parts[0]])
        
      
testDir=rootDir+'/Test'
testList=[]
count=0
for path, subdirs, files in os.walk(testDir):
  for name in files:
    if not name.endswith('.png'):
      continue         
    count = count +1  
    imPath=os.path.join(path, name); 
    image=cv2.imread(imPath,0) ### we want to read image as gray
    h,w=image.shape[:2]
    image=cv2.resize(image,(int(w*32/h),32))
    print('Test',count,image.shape)
    parts=name.split('_');   
    testList.append([image, parts[0]])

"""calculate some statistical information"""

maxWidth=0
maxLen=0
CharList=[]
for im,word in trainList:
  h,w=im.shape[:2]
  if maxLen<len(word):
    maxLen=len(word)
  if maxWidth<w:
    maxWidth=w
  CharList = list(set(CharList + list(word)))


for im,word in testList:
  h,w=im.shape[:2]
  if maxLen<len(word):
    maxLen=len(word)
  if maxWidth<w:
    maxWidth=w
  CharList = list(set(CharList + list(word)))

with open(rootDir+'/DataSet.pkl', 'wb') as f:
  cPickle.dump([trainList,testList,CharList,maxWidth,maxLen], f, pickle.HIGHEST_PROTOCOL)

"""Load Data from saved pickle file"""
