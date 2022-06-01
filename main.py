from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from NeuralNetwork import NeuralNetwork



#This function reads in all n images in catsfolder/ and dogsfolder/. 
#Each 64 x 64 image is reshaped into a length-4096 row vector. 
#These row vectors are stacked on top of one another to get a data matrix
#X of size n x 4096. We also generate a -1 label if the row vector corresponds
#to a cat image and a +1 label if the row vector corresponds to a dog image
#and stack these on top of one another to get a label vector y of length n.

def read_data():
    
    # get image filenames
    dogdir='virtualproject/DogCatClassifier/dogsfolder'
    dog_locs=[os.path.join(dogdir,fil) for fil in os.listdir(dogdir) if fil.endswith(".jpg")]

    catdir='virtualproject/DogCatClassifier/dogsfolder'
    cat_locs=[os.path.join(catdir,fil) for fil in os.listdir(catdir) if fil.endswith(".jpg")]

    num_cats = len(cat_locs)
    num_dogs = len(dog_locs)
    
    # initialize empty arrays
    X_cats = np.zeros((num_cats,64*64))
    X_dogs = np.zeros((num_dogs,64*64))
    y_cats = np.zeros((num_cats,1))
    y_dogs = np.zeros((num_dogs,1))
              
    #Load data, reshape into a 1D vector and set labels
    
    keep_track = 0

    for i in range(num_cats):
        img = cat_locs[i]
        im = io.imread(img)
        im = im.reshape(64*64)
        X_cats[i,:] = im
        y_cats[i] = 0
        keep_track += 1

    for i in range(num_dogs):
        img = dog_locs[i]
        im = io.imread(img)
        im = im.reshape(64*64)
        X_dogs[i,:] = im
        y_dogs[i] = 1.0
        keep_track += 1
    
    # combine both datasets
    X = np.append(X_cats,X_dogs,0)
    y = np.append(y_cats,y_dogs)
    
    return X, y 


def split_data(X,y,testpercent):
        
    [n, d] = X.shape
    
    ntest = int(round(n*(float(testpercent)/100)))
    ntrain = int(round(n - ntest))
        
    Xtrain = np.zeros((ntrain,d))
    Xtest = np.zeros((ntest,d))
    ytrain = np.zeros((ntrain,1))
    ytest = np.zeros((ntest,1))   
        
    Data = np.column_stack((X,y))
    Data = np.random.permutation(Data)
    
    for i in range(ntest):
        Xtest[i,:] = Data[i,0:d]
        ytest[i] = Data[i,d]
        
    for i in range(ntrain):
        Xtrain[i,:] = Data[i+ntest,0:d]
        ytrain[i] = Data[i+ntest,d]
        
    return Xtrain, ytrain, Xtest, ytest


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

#Read in data (look at one specific image by uncommenting show_image).
X,y = read_data()

#split into training and test data
Xtrain, ytrain, Xtest, ytest = split_data(X,y,20)

ytrain = np.transpose(ytrain)
#print(ytrain[0][1526])


#reducing the data
Xtrain_reduced, Xtest_reduced = reduce_data(Xtrain,Xtest,10)

#print(Xtrain_reduced.shape)
#print(ytrain)



input_vectors = Xtrain_reduced
targets = ytrain
learning_rate = 0.1

neural_network = NeuralNetwork(learning_rate, 10)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
plt.show() 
