# A Project in Neural Networks

In this repository, you’ll find several resources regarding multi-layer neural network. I just want to note 1) this is MLNN with backpropogation as the learning method, and 2) the activation function is the sigmoid function.

- In NNcasebycase.py, you’ll find several cases where feedforward, backprop, and gradient descent are hard coded rather than having a nice function for each. This is the best way to determine how to generalize the code. See below for a list of the cases I explore. Note, I tested my methods on logic gates. I have not yet explore vectorized images. In addition, I’ve tested various cases of hidden layers with a non-uniform number of nodes. I’ve found success in each case. The supporting functions aren’t well-documented, but I hope they are straight-forward. If not, please see myNN.py where I do document the necessary functions.

  0 hidden layers, one output
  
  0 hidden layers, two output
  
  1 hidden layer with two nodes each, one output
  
  1 hidden layer with two nodes each, two output
  
  2 hidden layer with two nodes each, one output
  
  2 hidden layer with two nodes each, two output
  
From the above code, I generalized my architecture in myNN.py. Each function is summarised and their parameters described. My method does converge in multiple cases. You will see at the bottom of the script, I’ve tested for outputs 1:4. In addition, you may see that I have 3 hidden layers with non-uniform nodes. The method does converge.

- In NN.py, I explore Python’s pre-built methods and include an example involving image classification for the hand-written digits. This code is well-documented as it is almost a direct copy from the sources listed. There are only slight changes.

I highly recommend neuralnetworksanddeeplearning.com for a resource if you attempting to understand neural nets yourself.  This source also goes into deep learning, an extension of neural networks.

Any input would be great as I am developing my programming and data science skills.  Thank you.
