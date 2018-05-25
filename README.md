# A Project in Neural Networks

In this repository, you’ll find several resources regarding multi-layer neural network. I just want to note 1) this is MLNN with backpropogation and simple gradient descent as the learning method, and 2) the activation function is the sigmoid function.

- In NNcasebycase.py, you’ll find several cases where feedforward, backprop, and gradient descent are hard coded rather than having a nice function for each. This is how I prefer to determine how to generalize the code. See below for a list of the cases I explore. Note, I tested my methods on logic gates. I have not yet explore vectorized images. In addition, I’ve tested various cases of hidden layers with a non-uniform number of nodes. I’ve found success in each case. The supporting functions aren’t well-documented (in this particular file). 

  - I have, however, created diagrams and a detailed the math for each case in graphs_galore.pdf.  For more information on the intuition, I highly recommend neuralnetworksanddeeplearning.com for a resource.  This source also goes into deep learning, an extension of neural networks.
  - In myNN.py, I generalize NNcasebycase.py and I do document the necessary functions.  My method does   converge in multiple cases. You will see at the bottom of the script, I’ve tested for outputs 1:4. In addition, you may see that I have 3 hidden layers with non-uniform number of nodes. The method does converge.
  - Cases
  
    #1) 0 hidden layers, one output
  
    #2) 0 hidden layers, two output
  
    #3) 1 hidden layer with two nodes each, one output
  
    #4) 1 hidden layer with two nodes each, two output
  
    #5) 2 hidden layer with two nodes each, one output
  
    #6) 2 hidden layer with two nodes each, two output
  
  From the above code, I generalized my architecture in myNN.py. Each function is summarised and their parameters described. 

Any input would be great as I am developing my programming and data science skills.  Thank you.  Questions are also welcome!
