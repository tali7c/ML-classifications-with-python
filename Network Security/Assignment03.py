#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:30:56 2019

@author: ali
"""

inputData=input('enter a number = ')
n=int(inputData)

fact=1
for item in range(2,n+1,1): # 2 to n number generate with gap 1
    fact *= item

print('The factorial of give numbers %d is %d'%(n, fact))