#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:30:56 2019

@author: ali
"""
import numpy as np

inputData=input('enter a list of number separated by space = ')
Datalist=np.array(inputData.split(' '), dtype='int32')

sumx=0
for item in Datalist:
    sumx += item

print('The sum of give numbers ', Datalist, ' is %d'%(sumx))