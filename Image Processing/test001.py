# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:32:17 2019

@author: EICT
"""

a=int(input('enter some data = '))

if(a%2==0):
    print('%d is a even number'%(a))
else:
    print('%d is a odd number'%(a))
    
    
    
    
str1=input('enter some values by , seprated = ')
parts=str1.split(',')
    
sumx=0
for item in parts:
    sumx = sumx + int(item)
    
print('the sum of given values is %d'%(sumx))    
    


n=int(input('enter a value for which a factorial is needed = '))
fact=1
for i in range(1,n+1,1):
    fact *= i

print('the factorial of given values %d is %d'%(n,fact))    
    