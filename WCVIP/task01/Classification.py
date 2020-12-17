# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:47:56 2020
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


rootDir='./Data/Train'
trainList=[]
for path, subdirs, files in os.walk(rootDir):
  for name in files:
    if not name.endswith('.png'):
        continue  
    
    imPath=os.path.join(path, name); 
    # print(imPath)
    parts=imPath.split(os.sep);   
    trainList.append([imPath, int(parts[-2])])
        
        
rootDir='./Data/Test'
testList=[]
for path, subdirs, files in os.walk(rootDir):
    for name in files:
        if not name.endswith('.png'):
            continue  
        
        imPath=os.path.join(path, name); 
        # print(imPath)
        parts=imPath.split(os.sep); 
        testList.append([imPath, int(parts[-2])])
  

class create_mnist_dataset(object):
    def __init__(self,batch_size):
        self.batch_size=batch_size
    def gen(self,dataList, phase='Train'):
        inps=[]
        labels=[]
        try:
          while 1:
              shuffle(dataList)
              for imPath, label in dataList:
                  # print(imPath)
                  tmp=np.zeros((1,1,10),dtype='uint8')
                  tmp[0,0,int(label)]=1
                  labels.append(tmp.copy())
                  image=cv2.imread(imPath,0) ### we want to read image as gray
                  image.shape
                  inps.append(np.expand_dims(image,axis=-1))
                  if len(labels)==self.batch_size:            
                      yield np.asarray(inps,dtype='float32'), np.asarray(labels,dtype='float32')
                      inps=[]
                      labels=[]
              if phase=='Test':
                  break
        except GeneratorExit:
          print("Generated Finished")

batch_size=256
# print(len(trainList))
dataset=create_mnist_dataset(batch_size)
traindata=dataset.gen(trainList)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


testdata=dataset.gen(testList,phase='Test')
vimages, vclasses = next( testdata)

# Fill out the subplots with the random images that you defined 
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.axis('off')
    plt.imshow(vimages[i].reshape(28,28))
    plt.subplots_adjust(wspace=0.5)
plt.show()

# Parameters
learning_rate = 0.01
training_iteration = 5000
display_step = 10


# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)




# Store layers weight & bias

initializer=tf.keras.initializers.HeNormal()  
# initializer=tf.keras.initializers.HeUniform()
# initializer = tf.initializers.orthogonal(gain=1.0) 
# initializer = tf.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')  
# initializer = tf.initializers.glorot_uniform()  
# initializer = tf.initializers.glorot_normal()  
# initializer = tf.initializers.RandomUniform(minval=-1.0, maxval=1.0)  
# initializer = tf.initializers.RandomNormal(mean=0, stddev=1.0)  

# Store layers weight & bias
weights = {
    'h1': tf.Variable(initializer([n_input, n_hidden_1]),trainable=True),
    'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2]),trainable=True),
    'out': tf.Variable(initializer([n_hidden_2, n_classes]),trainable=True)
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1]),trainable=True),
    'b2': tf.Variable(tf.zeros([n_hidden_2]),trainable=True),
    'out': tf.Variable(tf.zeros([n_classes]),trainable=True)
}

varList=[]
for key in weights:
  varList.append(weights[key])
for key in biases:
  varList.append(biases[key])


# Create model
def multilayerNN(x):
    x=tf.reshape(x,[-1,1,1,n_input])
    # print(tf.reduce_mean(x))
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1=tf.nn.leaky_relu(    layer_1,    alpha=0.2)
    # print(weights['h1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2=tf.nn.leaky_relu(    layer_2,    alpha=0.2)
    # print(layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # y=tf.nn.sigmoid(out_layer)
    # y=tf.clip_by_value(tf.nn.softmax(out_layer),0.01,0.99)
    y=tf.nn.softmax(out_layer)
    return y


optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95, epsilon=10.0**-7)
# optimizer = tf.keras.optimizers.SGD( learning_rate=learning_rate, momentum=0.0)

iteration=0
while iteration<training_iteration:
  iteration = iteration + 1 
  images, classes = next(traindata)
  # print(classes)
  images=tf.convert_to_tensor(images)
  images=(images-127.5)/127.5
  classes=tf.convert_to_tensor(classes)
  with tf.GradientTape(watch_accessed_variables=True) as tape:
    tape.watch(varList)
    y = multilayerNN(images)
    # print(y)
    loss = tf.reduce_mean(-classes*tf.math.log(y))

  gradients = tape.gradient(loss, varList)
  optimizer.apply_gradients(zip(gradients, varList))

  
  # Display loss per epoch step
  if iteration % display_step == 0:
      print("iteration:", '%04d' % (iteration), "cost={:.9f}".format(loss))

"""# ** Testing**"""

print(len(testList))
testdata=dataset.gen(testList,phase='Test')

totalN=0
totalP=0
while True:
  try:
    images, classes = next(testdata)
  except:
    break
  # print(classes)
  images=tf.convert_to_tensor(images)
  images=(images-127.5)/127.5
  classes=tf.convert_to_tensor(classes)
  y = multilayerNN(images)
  yclass=tf.argmax(y,axis=-1)
  classes=tf.argmax(classes,axis=-1)

  p=tf.reduce_sum(tf.cast( tf.equal(yclass,classes), dtype='float32'))
  totalP = totalP + p
  totalN = totalN + batch_size

  if iteration % display_step == 0:
      print("totalN:", '%04d' % (totalN), "totalP:", '%04d' % (totalP), "Accuracy={:.9f}".format(totalP*100/totalN))
            
            
