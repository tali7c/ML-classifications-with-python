# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:53:38 2020
conda create -n EICT -y
conda activate EICT
conda install tensorflow -y
conda install -c conda-forge opencv -y
#pip install opencv-python
conda install -c conda-forge python-wget -y
conda install matplotlib -y
pip install wget
@author: ALI
"""

import os
import numpy as np
import cv2 
import wget

rootDir='./Data'
os.makedirs(rootDir, exist_ok=True)    
        
os.chdir(rootDir)


url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
wget.download(url)
url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
wget.download(url)
url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
wget.download(url)
url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
wget.download(url)



image_size = 28
import gzip
fx = gzip.open('train-images-idx3-ubyte.gz','r')



fx.read(16)
fy = gzip.open('train-labels-idx1-ubyte.gz','r')
fy.read(8)
count=0
while 1:
    labBuff=fy.read(1)
    if not labBuff:
        break
    label = np.frombuffer(labBuff, dtype=np.uint8)[0]
    buf = fx.read(image_size * image_size )
    data = np.frombuffer(buf, dtype=np.uint8)
    image = data.reshape(image_size, image_size)
    parentDir='Train'+os.sep+str(label)
    os.makedirs(parentDir, exist_ok=True)    
    cv2.imwrite(parentDir+os.sep+str(count)+'.png',image)  
    count += 1


fx = gzip.open('t10k-images-idx3-ubyte.gz','r')
fx.read(16)
fy = gzip.open('t10k-labels-idx1-ubyte.gz','r')
fy.read(8)
count=0
while 1:
    labBuff=fy.read(1)
    if not labBuff:
        break
    label = np.frombuffer(labBuff, dtype=np.uint8)[0]
    buf = fx.read(image_size * image_size )
    data = np.frombuffer(buf, dtype=np.uint8)
    image = data.reshape(image_size, image_size)
    parentDir='Test'+os.sep+str(label)
    os.makedirs(parentDir, exist_ok=True)    
    cv2.imwrite(parentDir+os.sep+str(count)+'.png',image)  
    count += 1
