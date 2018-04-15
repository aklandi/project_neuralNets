import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import neural_network
from sklearn import metrics

#inspiration: 
# (1) https://www.datacamp.com/community/tutorials/machine-learning-python#gs.geiQ7Ic
# (2) http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html

# We want to load the dataset of digits
digits = datasets.load_digits()
# We want to separate the images from the rest of the info
images = digits.images
# separate the data of the images (basically the images vectorized)
X = digits.data
# Separate the target values from the rest of the info
y = digits.target
# partition the data into training data and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y)
# scale the data because multilayer preceptron may not converge
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

#scaler = preprocessing.StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# The neural network "architecture" is created here.  We tell it to create 3 hidden layers
# with 45 nodes each
mlp = neural_network.MLPClassifier(hidden_layer_sizes = (20,20))
# now we send our training data in to have the weights and thresholds trained
mlp.fit(X_train, y_train)
# These are our predictions based on the trained network
predictions = mlp.predict(X_test)
# the confusion matrix gives the counts of how many of the predictions corresponds
# to the test output
print(metrics.confusion_matrix(y_test, predictions))
# precision and recall are provided to determine the 
# accuracy of our predictions
print(metrics.classification_report(y_test,predictions))

# Now we want to visualize what is going on... hence why I went with image data
 
# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))
# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# For just 64 images; note there are more than 64 images in the test set (450).
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(X_test[i].reshape(8,8), cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(y_test[i]))
# Show the plot
plt.show()
