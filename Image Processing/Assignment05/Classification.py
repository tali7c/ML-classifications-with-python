# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:53:38 2019
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
        
        parts=imPath.split(os.sep);   
        trainList.append([imPath, int(parts[-2])])
        
        
rootDir='./Data/Test'
testList=[]
for path, subdirs, files in os.walk(rootDir):
    for name in files:
        if not name.endswith('.png'):
            continue  
        
        imPath=os.path.join(path, name); 
        
        parts=imPath.split(os.sep);   
        testList.append([imPath, int(parts[-2])])
  

class create_mnist_dataset(object):
    def __init__(self,batch_size):
        self.batch_size=batch_size
    def gen(self,dataList, phase='Train'):

        inps=[]
        labels=[]
        while 1:
            shuffle(dataList)
            for imPath, label in dataList:
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


batch_size=256
dataset=create_mnist_dataset(batch_size)
traindata=dataset.gen(trainList)
validdata=dataset.gen(testList,phase='Test')



vimages, vclasses = next(validdata)

# Fill out the subplots with the random images that you defined 
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.axis('off')
    plt.imshow(vimages[i].reshape(28,28))
    plt.subplots_adjust(wspace=0.5)

plt.show()

# Parameters
learning_rate = 0.001
training_iteration = 1000
display_step = 10


# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)




# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])



# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x):

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#    layer_1actv=tf.nn.leaky_relu(    layer_1,    alpha=0.2)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#    layer_2actv=tf.nn.leaky_relu(    layer_2,    alpha=0.2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer




# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)




# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    iteration=0
    while iteration<training_iteration:
        
        iteration += 1
        
        
        
        images, classes = next(traindata)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_op, loss_op],
        feed_dict={X: np.reshape(images,(-1,n_input)),
                  Y: np.reshape(classes,(-1,10))})
        
        # Display logs per epoch step
        if iteration % display_step == 0:
            print("iteration:", '%04d' % (iteration), "cost={:.9f}".format(c))
            
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: np.reshape(vimages,(-1,n_input)), Y: np.reshape(vclasses,(-1,10))}))
            
            
