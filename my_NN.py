import numpy as np
import matplotlib.pyplot as plt

#
# param: x, can be a value or a numpy array
# 
# return: the sigmoid value of x
#
def sigmoid(x):
    
    return 1/(1+np.exp(-x))
    
#
# param: input, a list of list, containing training data
#
# return: a numpy array of the input concatonated with
#         the bias value 1 for each training sample.
#         That is, [1,0] will be returned as [1, 1, 0].
#
def input_setup(input):
    
    n_row, n_col = np.shape(input)
    one = np.ones([n_row,1])    
    X = np.hstack([one,input])
    
    return X
    
#
# param: xi, a numpy 1d array training set 
# param: wi, a numpy 2d array of weights - can be a single vector or the whole set
#
# return: the dot product of xi and wi if xi is a vector; the matrix multiplication
#         of xi and wi if wi is a 2d array
def weighted_sum(xi, wi):
    
    if wi.ndim == 2:
        
        n,m = np.shape(wi)
        d = np.zeros(m)
        
        for j in range(m):
            
            for i in range(n):
                
                d[j] += xi[i]*wi[i,j]
    
    else:
        
        d = 0
    
        for i in range(len(xi)):
        
            d += xi[i]*wi[i]
        
    return d
    
#
# param: input, a 2d numpy array, the whole training set plus bias
# param: n_nodes, a list containing the number of nodes in each hidden layer
# param: n_hidden, a scalar of how many hidden layers
# param: n_output, a scalar of how many output nodes there are
#
# return: a list of numpy arrays for weights for each layer
#
def initialize_weights(input, n_nodes, n_hidden, n_output):
    
    n_row, n_col = np.shape(input)
    
    w = []
    
    # need to consider three cases: 
    #       no hidden layers, one hidden layer, and many hidden layers
    if n_hidden == 0:
        
        # I use the standard normal distribution for initialization
        weights = np.random.normal(0,1,[n_col, n_output])
        w.append(weights)
        
    elif n_hidden == 1:
        
        weights = np.random.normal(0,1,[n_col, n_nodes[0]])
        w.append(weights)
        weights = np.random.normal(0,1,[n_nodes[0]+1, n_output])
        w.append(weights)
        
    else:
        
        weights = np.random.normal(0,1,[n_col, n_nodes[0]])
        w.append(weights)
        
        for layers in range(0, n_hidden-1):
            
            weights = np.random.normal(0,1,[n_nodes[layers] + 1, n_nodes[layers+1]])
            w.append(weights)
            
        weights = np.random.normal(0,1,[n_nodes[n_hidden-1]+1, n_output])
        w.append(weights)
    
    return w
    
#
# param: x, a 2d or 1d array
# param: y, has the same dimensions of x
#
# return: a 2d array that is the element wise subtraction of x and y
#
def sub(x,y):
    
    # this is for two 2d arrays of the same shape
    if y.ndim == 2:
        
        n,m = np.shape(y)
        s = np.zeros([n,m])
        
        for i in range(n):
            
            for j in range(m):
                
                s[i,j] = x[i,j] - y[i,j]
          
    # I primarily need this function for 1d arrays that are
    # the same length, but Python won't subtract (,nL) and (nL,)
    else: 
        
        n = len(x)
        s = np.zeros([n,1])
    
        for i in range(n):
        
            s[i] = x[i] - y[i]
        
    return s
    
    
#
# param: x, a numpy 1d array of length n
# param: y, a numpy 1d array of length n
#
# return: the element-wise product between x and y 
#    
def hadamard(x,y):
    
    # element-wise multiplication
    # primarily needed for 1d arrays that are the same length
    # but Python won't multiply (,nL) and (nL,)
    n = len(x)
    prod = np.zeros(n)
    
    for i in range(n):
        
        prod[i] = x[i]*y[i]
        
    return prod
    
#
# param: xi, a 1d numpy array of a single training vector plus bias
# param: weights, a list of numpy arrays containing the weights for each layer
# param: n_hidden, how many hidden layers there are
# param: n_output, how many output nodes there are
# param: predict, a boolean, default is false.  
#           When true, we simply return the output activation.
#           When false, we return the node activations for all layers, including output.  
#
# return: a list of 1d numpy arrays, the node activations for all hidden layers and the output layer
#
def feedforward(xi, weights, n_hidden, n_output, predict = False):
    
    # again, we consider three cases: 
    #       no hidden layers, one hidden layer, or multiple hidden layers
    if n_hidden == 0:
        
        _, m = weights[0].shape
        node_activ = np.zeros([1, m])
        node_activ = [sigmoid(np.matmul(xi, weights[0]))]
        
    elif n_hidden == 1:
        
        node_activ = []
        
        hidden = sigmoid(np.matmul(np.array([xi]), weights[0])) 
        hidden = np.hstack([ [[1]], hidden])
        node_activ.append(hidden)

        hidden = sigmoid(np.matmul(hidden, weights[1])) 
        node_activ.append(hidden)

    else:
        
        node_activ = []
        
        hidden = sigmoid(np.matmul(np.array([xi]), weights[0])) 
        hidden = np.hstack([ [[1]], hidden])
        node_activ.append(hidden)
        
        for i in range(1, n_hidden):
            
            hidden = sigmoid( np.matmul(hidden, weights[i]) )
            hidden = np. hstack([ [[1]], hidden])
            node_activ.append(hidden)
            
        hidden = sigmoid(np.matmul(hidden,weights[n_hidden]))
        node_activ.append(hidden)
        
    if predict == True:
        
        return node_activ[n_hidden]
        
    else:

        return node_activ

#
# param: X, a test set, preprocessed with bias term 1
# param: n_samples, the number of samples in the test set
# param: predicted_weights, a list of numpy arrays containing the weights given
#           by our trained system
# param: n_output, the number of output nodes
#
# return: the output of our test set
#
def test(X, n_samples, predicted_weights, n_hidden, n_output):
    
    predict = np.zeros([n_samples, n_output])
    for j in range(n_samples):
        
        xi = X[j,:]
        A = feedforward(xi, predicted_weights, n_hidden, n_output, predict = True)
        predict[j,:] = A
        
    print(predict)
    
#
# param: train_data, a list of lists containing the input data and the truth to
#           train our weights on
# param: nn_iter, the number of Neural net iterations
# param: step_size, a learning rate for gradient descent
# param: n_hidden, how many hidden layers requested
# param: n_output, the number of output nodes needed
# param: n_nodes, a list of the number of nodes per hidden layer
#
# return: a list of numpy arrays containing the trained weights per layer
#
def NN(train_data, num_samples, nn_iter, step_size, n_hidden, n_output, n_nodes):

    X = input_setup(train_data[0])
    weights = initialize_weights(X, n_nodes, n_hidden, n_output)
    
    for i in range(nn_iter):
        
        for j in range(num_samples):
    
            xi = X[j,:]
            ti = train_data[1][j]
            activ = feedforward(xi, weights, n_hidden, n_output)
            rate = backprop(xi, ti, activ, weights, n_hidden, n_output)
            
            for k in range(n_hidden+1):
                
                weights[k] = sub(weights[k], step_size*rate[k])
        
    return weights

#
# param: xi, one training example, a 1d vector or 1d list of values
# param: ti, the truth associated with xi, can be a scalar or a list of values
# param: node_activ, a list of arrays of the node activations per layer including the output layer
# param: weights, a list of arrays for the weights per layer
# param: n_hidden, the number of hidden layers, not including the input and output layers
# param: n_output, the number of outputs
#
# return: a list of 2d numpy arrays containing the change in weights for each layer
#
def backprop(xi, ti, node_activ, weights, n_hidden, n_output):
    
    #
    # There are multiple cases to consider.  First,
    # whether there are 0 hidden layers, 1 hidden layer, or many hidden layers.
    # And, within these, we need to check whether there is one output or many outputs.
    # Each case changes the way delta acts on weights and the derivative of each hidden
    # activation.
    
    if n_hidden == 0:
        
        delta = (node_activ[0] - ti)*node_activ[0]*(1-node_activ[0])
        
        if n_output == 1:
            
            rate = [delta*xi]
            
        else:
            
            rate_temp = delta[0]*xi
            
            for i in range(1,n_output):
                
                rate_temp = np.vstack([rate_temp, delta[i]*xi])
        
            rate = [rate_temp.T]
            
    elif n_hidden == 1:
        
        delta2 = [(node_activ[1][0] - ti)*node_activ[1][0]*(1-node_activ[1][0])]
        
        if n_output == 1:
            
            delta1 = delta2[0]*(node_activ[0][0][1:]*(1-node_activ[0][0][1:])*weights[1][1:].T)
            
        else:
        
            delta1 = 0
            
            for i in range(n_output):
                
                delta1 += delta2[0][i]*(node_activ[0][0][1:]*(1-node_activ[0][0][1:])*weights[1][1:,i])
                
            delta1 = [delta1]
                
        rate2 = np.matmul(np.array([node_activ[0][0]]).T, delta2)
        rate1 = np.matmul(np.array([xi]).T, delta1)
        rate = [rate1,rate2] 
            
    else:
        
        delta = []
        rate = []
        
        delta = [(node_activ[n_hidden][0] - ti)*node_activ[n_hidden][0]*(1-node_activ[n_hidden][0])] + delta
        
        if n_output == 1:
            
            delta = [delta[0]*hadamard(node_activ[n_hidden-1][0][1:]*(1-node_activ[n_hidden-1][0][1:]),weights[n_hidden][1:])] + delta
            
            count = n_hidden - 2
            
            while count > -1 :
                delta_temp = 0
                for i in range(n_nodes[count+1]):
            
                    delta_temp += delta[0][i]*node_activ[count][0][1:]*(1-node_activ[count][0][1:])*weights[count+1][1:,i]
            
                delta = [delta_temp] + delta
                count -= 1
            
        else:
            
            delta_temp = 0
            for i in range(n_output):
        
                delta_temp += delta[0][i]*node_activ[n_hidden-1][0][1:]*(1-node_activ[n_hidden-1][0][1:])*weights[n_hidden][1:,i]
        
            delta = [delta_temp] + delta   
               
                
            count = n_hidden - 2
            while count > -1 :
                delta_temp = 0
                for i in range(n_nodes[count+1]):
            
                    delta_temp += delta[0][i]*node_activ[count][0][1:]*(1-node_activ[count][0][1:])*weights[count+1][1:,i]
            
                delta = [delta_temp] + delta
                count -= 1
                
        for k in range(n_hidden, 0, -1):
            
            hidden = node_activ[k-1][0]
            
            d = delta[k]
            
            rate = [np.matmul(np.array([hidden]).T,np.array([d]))] + rate
            
        rate = [np.matmul(np.array([xi]).T,np.array([delta[0]]))] + rate
     
    return rate
    
# main program
input = [[1,0],[0,1],[0,0],[1,1]]
truth1 = [1,1,1,0] #NAND
truth2 = [[1,1],[1,1],[1,0],[0,1]] #NAND and OR
truth3 = [[1,1,1],[1,1,1],[1,0,0],[0,1,0]] #NAND, OR, XOR
truth4 = [[1,1,1,0],[1,1,1,0],[1,0,0,1],[0,1,0,1]] #NAND, OR, XOR, XNOR

#concatonate the input and the truth
num_samples, num_features = np.shape(input)
train_data = [input,truth1]
#parameters to be set prior to function call
nn_iter = 5000
step_size = 0.3
n_hidden = 0
n_output = 1
n_nodes = [5,3,4]

# learning method for the weights of our s
w = NN(train_data, num_samples, nn_iter, step_size, n_hidden, n_output, n_nodes)
test(input_setup(input), num_samples, w, n_hidden, n_output)
