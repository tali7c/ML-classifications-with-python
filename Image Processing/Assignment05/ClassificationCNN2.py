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
import pickle
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


batch_size=10
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
training_iteration = 100
display_step = 1


# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = [28,28,1] # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)




# tf Graph input
X = tf.placeholder("float", [None, n_input[0], n_input[1], n_input[2]])
Y = tf.placeholder("float", [None, n_classes])


weights1,biases1=pickle.load(open('weights.pkl', 'rb'))
weights = {
    'cnn1': tf.get_variable("cnn1",initializer=weights1['cnn1']),
    'cnn2': tf.get_variable("cnn2",initializer=weights1['cnn2']),
    
    
    'cnn3': tf.get_variable("cnn3",initializer=weights1['cnn3']),
    'cnn4': tf.get_variable("cnn4",initializer=weights1['cnn4']), 
    
    'cnn5': tf.get_variable("cnn5",initializer=weights1['cnn5']),
                       
    'h1': tf.get_variable("h1",initializer=weights1['h1']),
    'h2': tf.get_variable("h2",initializer=weights1['h2']),
    'out': tf.get_variable("out",initializer=weights1['out'])
}
biases = {
    'cb1': tf.get_variable("cb1",initializer=biases1['cb1']),
    'cb2': tf.get_variable("cb2",initializer=biases1['cb2']),
    'cb3': tf.get_variable("cb3",initializer=biases1['cb3']),
    'cb4': tf.get_variable("cb4",initializer=biases1['cb4']),
    'cb5': tf.get_variable("cb5",initializer=biases1['cb5']),
    
    'b1': tf.get_variable("b1",initializer=biases1['b1']),
    'b2': tf.get_variable("b2",initializer=biases1['b2']),
    'out': tf.get_variable("outb",initializer=biases1['out']),
}

# Create model
def multilayer_perceptron(x):

    cnn1_out=tf.add( tf.nn.conv2d( x, weights['cnn1'], [1,1,1,1], padding='VALID'), biases['cb1'])
    cnn1_actv=tf.nn.leaky_relu(cnn1_out, alpha=0.2)
    
    cnn2_out=tf.add( tf.nn.conv2d( cnn1_actv, weights['cnn2'], [1,1,1,1], padding='VALID'), biases['cb2'])
    cnn2_actv=tf.nn.leaky_relu(cnn2_out, alpha=0.2)
    
    
    pool1=tf.nn.max_pool(cnn2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    
    cnn3_out=tf.add( tf.nn.conv2d( pool1, weights['cnn3'], [1,1,1,1], padding='VALID'), biases['cb3'])
    cnn3_actv=tf.nn.leaky_relu(cnn3_out, alpha=0.2)
    cnn4_out=tf.add( tf.nn.conv2d( cnn3_actv, weights['cnn4'], [1,1,1,1], padding='VALID'), biases['cb4'])
    cnn4_actv=tf.nn.leaky_relu(cnn4_out, alpha=0.2)
    
    pool2=tf.nn.max_pool(cnn4_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    
    cnn5_out=tf.add( tf.nn.conv2d( pool2, weights['cnn5'], [1,1,1,1], padding='VALID'), biases['cb5'])
    cnn5_actv=tf.nn.leaky_relu(cnn5_out, alpha=0.2)
    
    
    flatten1=tf.reshape(cnn5_actv,(-1,64*2*2))


    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(flatten1, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer




# Construct model
logits = multilayer_perceptron(X)

VarList=tf.trainable_variables()

regularizer = tf.contrib.layers.l2_regularizer(10.0**-6)                            
reg_term = tf.contrib.layers.apply_regularization(regularizer,VarList)

# Define loss and optimizer
loss_op = reg_term + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
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
        _, c = sess.run([train_op, loss_op], feed_dict={X: images,
                                                        Y: np.reshape(classes,(-1,10))})
        
        # Display logs per epoch step
        if iteration % display_step == 0:
            print("iteration:", '%04d' % (iteration), "cost={:.9f}".format(c))
            
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: vimages, Y: np.reshape(vclasses,(-1,10))}))
            

    # Run optimization op (backprop) and cost op (to get loss value)
    weights,biases = sess.run([weights, biases])
    pickle.dump([weights,biases], open('weights.pkl', 'wb'))



