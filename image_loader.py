"""
image_loader
~~~~~~~~~~~~
A library to load the dogs and cats image data.
"""

#### Libraries
# Standard library
import pickle
import gzip
import os

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float


def read_data():
    """This function reads in all n images in catsfolder/ and dogsfolder/. 
    Each 64 x 64 image is reshaped into a length-4096 row vector. 
    These row vectors are stacked on top of one another to get a data matrix
    X of size n x 4096. We also generate a -1 label if the row vector corresponds
    to a cat image and a +1 label if the row vector corresponds to a dog image
    and stack these on top of one another to get a label vector y of length n.
    """
    
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
    y_cats = np.zeros((num_cats))
    y_dogs = np.zeros((num_dogs))
              
    #Load data, reshape into a 1D vector and set labels
    
    keep_track = 0

    for i in range(num_cats):
        img = cat_locs[i]
        im = img_as_float(io.imread(img, True))
        im = im.reshape(64*64)
        X_cats[i,:] = im
        y_cats[i] = 0
        keep_track += 1

    for i in range(num_dogs):
        img = dog_locs[i]
        im = img_as_float(io.imread(img, True))
        im = im.reshape(64*64)
        X_dogs[i,:] = im
        y_dogs[i] = 1
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
    ytrain = np.zeros((ntrain))
    ytest = np.zeros((ntest))   
        
    Data = np.column_stack((X,y))
    Data = np.random.permutation(Data)
    
    for i in range(ntest):
        Xtest[i,:] = Data[i,0:d]
        ytest[i] = Data[i,d]
        
    for i in range(ntrain):
        Xtrain[i,:] = Data[i+ntest,0:d]
        ytrain[i] = Data[i+ntest,d]

    #training_data = list(zip(Xtrain, ytrain))
    #testing_data = list(zip(Xtest, ytest))
    Xtrain = np.transpose(Xtrain)
    Xtest = np.transpose(Xtest)


        
    return Xtrain, ytrain, Xtest, ytest



def show_image(X, i):
    """This function takes in an n x 4096 data matrix X and an index i. It extracts
    the ith row of X and displays it as a grayscale 64 x 64 image.
    """

    #select image
    image = X[:,i]
    #reshape make into a square
    image = image.reshape((64,64))
    #display the image
    plt.imshow(image,'gray')


def y_vectorization(y_data):
    """Returns a 2-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert the class (dog or cat)
    into a corresponding desired output from the neural network.
    """
    if y_data == 1:          # If the image corresponds to a dog
        y_data = np.zeros((2, 1))
        y_data[0] = 1.0
    else:
        y_data = np.zeros((2, 1))
        y_data[1] = 1.0
    return y_data



def load_data():
    """
    """

    #Read in data (look at one specific image by uncommenting show_image).
    X,y = read_data()
    #split into training and test data
    training_data, testing_data = split_data(X,y,20)

    training_input = training_data[0]
    training_results = [y_vectorization(y) for y in training_data[1]]

    testing_input = testing_data[0]
    testing_results = [y_vectorization(y) for y in testing_data[1]]



    training_data = list(zip(training_input, training_results))
    testing_data = list(zip(testing_input, testing_results))

    return (training_data, testing_data, testing_data)

