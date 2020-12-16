#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:06:51 2019

@author: ali
"""

# plot a curve
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 100)
y=np.sin(x)
#create a new figure window
fig=plt.figure()
#plot on the figure
plt.plot(x, x, label='linear')
plt.plot(x, y, label='sin')

#set title and labels
plt.title('matplotlibExamle01')
plt.xlabel('x')
plt.ylabel('y')

#show legend
plt.legend()

# Show the plot
plt.show()
# save the fig
plt.savefig('matplotlibExamle01.png')




# limit the x and y axis
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 100)
y=np.sin(x)
#create a new figure window
fig=plt.figure()

#plot on the figure
plt.plot(x, np.cos(x), label='cos')
plt.plot(x, y, label='sin')
plt.plot(x, np.tan(x), label='tan')
#set title and labels
plt.title('matplotlibExamle02')

plt.xlabel('x')
plt.ylabel('y')

#limit the axis
plt.xlim([-5,5])
plt.ylim([0,5])

#show legend
plt.legend()

# Show the plot
plt.show()
# save the fig
plt.savefig('matplotlibExamle02.png')


#creating multiple subplot
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10, 10, 100)
y=np.sin(x)
#create a new figure window
fig=plt.figure() # plt.clf() can be used to clear the plt figure

#plot on the figure
plt.subplot(1,3,1) # plt.subplot(rows,cols, id)
plt.plot(x, np.cos(x), label='cos')
plt.title('cos(x)')

plt.subplot(1,3,2)
plt.plot(x, y, label='sin')
plt.title('sin(x)')

plt.subplot(1,3,3)
plt.plot(x, np.tan(x), label='tan')
plt.title('tan(x)')
#set title and labels
plt.suptitle('matplotlibExamle03') # to create single title for the all subplot
 
plt.xlabel('x')
plt.ylabel('y')

#limit the axis
plt.xlim([-5,5])
plt.ylim([0,5])

#show legend
plt.legend()

# Show the plot
plt.show()
# save the fig
plt.savefig('matplotlibExamle03.png')



