#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:30:56 2019

@author: ali
"""

import numpy as np  #conda install numpy
import pandas as pd #conda install pandas # read csv file
import matplotlib.pyplot as plt #conda install -c anaconda matplotlib
from sklearn.model_selection import train_test_split #conda install -c anaconda scikit-learn
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


colnames=['duration', 'protocol_type',' service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','label']


#Reading data from CSV file
kdd_df = pd.read_csv("/mnt/BackUp/InSync/shared data/ML-classifications-with-python/Network Security/data/kddcup.data",  names=colnames, header=None)

freq=kdd_df['label'].value_counts()

d1=kdd_df.loc[kdd_df['label'] == 'smurf.'].sample(n=10000)
d2=kdd_df.loc[kdd_df['label'] == 'neptune.'].sample(n=10000)
d3=kdd_df.loc[kdd_df['label'] == 'normal.'].sample(n=10000)
d4=kdd_df.loc[kdd_df['label'] == 'satan.'].sample(n=10000)


kdd_df1=pd.concat([d1,d2,d3,d4], axis=0,ignore_index=True)

newKDD=pd.DataFrame()

for i,col in enumerate(kdd_df1.columns):
    
   
    tt=kdd_df1[col]
    # print(i,col,tt.dtypes)
    if tt.dtypes==object and not col=='label':         
        print(i,col,tt.dtypes)
        one_hot = pd.get_dummies(kdd_df1[col])
        newKDD=pd.concat([newKDD,one_hot], axis=1)
    else:
        newKDD=pd.concat([newKDD,tt], axis=1)
    
        
    



# View the first 5 rows of the data
print(newKDD.head())


#Defining data and label
X = newKDD.iloc[:, 0:115]
y = newKDD.iloc[:, 115]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))


sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# plt.subplot(2,1,1)
# markers = ('s', 'x', 'o')
# colors = ('red', 'blue', 'lightgreen')
# cmap = ListedColormap(colors[:len(np.unique(y_train))])
# for idx, cl in enumerate(np.unique(y_train)):
#     plt.scatter(x=X_train_std[y_train == cl, 0], y=X_train_std[y_train == cl, 1],
#                c=cmap(idx), marker=markers[idx], label=cl)
    

# plt.subplot(2,1,2)
# markers = ('s', 'x', 'o')
# colors = ('red', 'blue', 'lightgreen')
# cmap = ListedColormap(colors[:len(np.unique(y_train))])
# for idx, cl in enumerate(np.unique(y_train)):
#     plt.scatter(x=X_train_std[y_train == cl, 0], y=X_train_std[y_train == cl, 3],
#                c=cmap(idx), marker=markers[idx], label=cl)
    
        
    

    
#Multi Layer Perceptron Classifier
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic',  learning_rate='constant', learning_rate_init=0.001, max_iter=100, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1)

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic',  learning_rate='constant', learning_rate_init=0.001, max_iter=1000, alpha=1e-4,
#                      solver='adam', verbose=10, tol=1e-4, random_state=1)


mlp.fit(X_train, y_train)
print("MLP Training set score: %f" % mlp.score(X_train, y_train))
print("MLP Test set score: %f" % mlp.score(X_test, y_test))


#K Nearnest Neighbour
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("knn Training set score: %f" % knn.score(X_train, y_train))
print("knn Test set score: %f" % knn.score(X_test, y_test))


# #Support Vector Machine
# from sklearn.svm import SVC

# svmRBF=SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
# svmRBF.fit(X_train, y_train)
# print("SVM Training set score: %f" % svmRBF.score(X_train, y_train))
# print("SVM Test set score: %f" % svmRBF.score(X_test, y_test))



#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

decisionTree=DecisionTreeClassifier(max_depth=5)
decisionTree.fit(X_train, y_train)
print("decisionTree Training set score: %f" % decisionTree.score(X_train, y_train))
print("decisionTree Test set score: %f" % decisionTree.score(X_test, y_test))



#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

randomForest=RandomForestClassifier(max_depth=5, n_estimators=100, max_features=115)
randomForest.fit(X_train, y_train)
print("randomForest Training set score: %f" % randomForest.score(X_train, y_train))
print("randomForest Test set score: %f" % randomForest.score(X_test, y_test))



# # print all score
# print("MLP Training set score: %f" % mlp.score(X_train, y_train))
# print("MLP Test set score: %f" % mlp.score(X_test, y_test))
# print("KNN Training set score: %f" % knn.score(X_train, y_train))
# print("KNN Test set score: %f" % knn.score(X_test, y_test))
# print("SVM Training set score: %f" % svmRBF.score(X_train, y_train))
# print("SVM Test set score: %f" % svmRBF.score(X_test, y_test))
# print("randomForest Training set score: %f" % randomForest.score(X_train, y_train))
# print("randomForest Test set score: %f" % randomForest.score(X_test, y_test))
# print("decisionTree Training set score: %f" % decisionTree.score(X_train, y_train))
# print("decisionTree Test set score: %f" % decisionTree.score(X_test, y_test))
