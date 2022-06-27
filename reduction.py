"""
reduction
~~~~~~~~~~~~
A library to reduce dimentionality by PCA.
"""

#### Libraries
# Standard library

# Third-party libraries
import numpy as np


#This code implements the PCA exactly as in MATLAB so as to be consistent.
#It takes in an n x d data matrix X and returns a d x d orthonormal matrix pcaX. 
#Each column of pcaX contains a basis vector, sorted by decreasing variance.

def pca(X):
    covX = np.cov(X,rowvar=False)
    [Lambda,Vtranspose] = np.linalg.eig(covX)
    neworder = np.argsort(-abs(Lambda))
    pcaX = Vtranspose[:,neworder]
    pcaX = pcaX.real

    return pcaX

#This function takes in a training data matrix Xtrain and uses
#it to compute the PCA basis and a sample mean vector. 
#It also takes in a test data matrix Xtest and a dimension k. 
#It first centers the data matrices Xtrain and Xtest by subtracting the
#Xtrain sample mean vector from each of their rows. It then uses the 
#top-k vectors in the PCA basis to project the centered Xtrain and Xtest
#data matrices into a k-dimensional space, and outputs
#the resulting data matrices as Xtrain_reduced and Xtest_reduced.
def reduce_data(Xtrain,Xtest,k):
    VT = pca(Xtrain)
    VTk = VT[:,:k]
    [ntrain,d] = Xtrain.shape
    [ntest,d] = Xtest.shape
    mean_train = np.average(Xtrain,axis = 0)
    onesvector_train = np.ones((ntrain,))
    Xtrain_center = Xtrain - np.outer(onesvector_train,mean_train)
    Xtrain_reduced = np.matmul(Xtrain_center,VTk)
    onesvector_test = np.ones((ntest,))
    Xtest_center = Xtest - np.outer(onesvector_test,mean_train)
    Xtest_reduced = np.matmul(Xtest_center,VTk)
        
    return Xtrain_reduced, Xtest_reduced