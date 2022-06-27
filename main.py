from turtle import shape
from DogCatClassifier.DeepNeuralNetwork import *
from DogCatClassifier.image_loader import *





# Read in data (look at one specific image by uncommenting show_image).
X,y = read_data()

# Split into training and test data
Xtrain, ytrain, Xtest, ytest = split_data(X,y,20)

# Initialize the neural network and train it
nn = DeepNeuralNetwork(0.5, [4096, 400, 2], "sigmoid")
print(nn.feedforward(np.transpose([Xtrain[:,0]])))





