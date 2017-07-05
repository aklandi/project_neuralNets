import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    
    return 1/(1+np.exp(-x))
    
def input_setup(input):
    
    n_row, n_col = np.shape(input)
    one = np.ones([n_row,1])    
    X = np.hstack([one,input])
    
    return X
    
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
    
def initialize_weights(input, n_nodes, n_hidden, n_output):
    
    n_row, n_col = np.shape(input)
    
    w = []
    
    if n_hidden == 0:
        
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

def sub(x,y):
    
    if y.ndim == 2:
        
        n,m = np.shape(y)
        s = np.zeros([n,m])
        
        for i in range(n):
            
            for j in range(m):
                
                s[i,j] = x[i,j] - y[i,j]
                
    else: 
        
        n = len(x)
        s = np.zeros([n,1])
    
        for i in range(n):
        
            s[i] = x[i] - y[i]
        
    return s
    
def hadamard(x,y):
    
    n = len(x)
    prod = np.zeros(n)
    
    for i in range(n):
        
        prod[i] = x[i]*y[i]
        
    return prod
    
# Example with no hidden layers and one output    
# Testing on NAND    
input = [[1,0],[0,1],[0,0],[1,1]]
truth = [1,1,1,0]

num_samples, num_features = np.shape(input)
nn_iter = 2000
step_size = 0.2

n_nodes = [1]
n_hidden = 0
n_output = 1

X = input_setup(input)
weights = initialize_weights(X, n_nodes, n_hidden, n_output)

for i in range(nn_iter):
    
    for j in range(num_samples):

        xi = X[j,:]
        
        #feedforward
        o1 = sigmoid(weighted_sum(xi,weights[0]))
        
        #backprop
        delta = (o1 - truth[j])*o1*(1-o1)
        derivW = delta*xi
        
        #grad descent
        weights[0] = sub(weights[0], step_size*derivW)
    
#test
predict = sigmoid(np.matmul(X,weights)); print(predict)   
     
# Example with no hidden layers and two outputs
# testing NAND and OR
input = [[1,0],[0,1],[0,0],[1,1]]
truth = [[1,1],[1,1],[1,0],[0,1]]

num_samples, num_features = np.shape(input)
nn_iter =1000
step_size = 0.3

n_nodes = [1]
n_hidden = 0
n_output = 2

X = input_setup(input)
weights = initialize_weights(X, n_nodes, n_hidden, n_output)

for i in range(nn_iter):
    
    for j in range(num_samples):

        xi = X[j,:]
        
        #feedforward 
        o = sigmoid(weighted_sum(xi, weights[0]))
        
        #backprop
        diff = (o - truth[j])*o*(1-o)
        derivW = np.vstack([diff[0]*xi, diff[1]*xi])
        
        #grad descent
        weights[0] -= step_size*derivW.T
    
#test
predict = sigmoid(np.matmul(X,weights[0])); print(predict) 

#Example with one hidden layer with 2 nodes and one output
#NAND again
input = [[1,0],[0,1],[0,0],[1,1]]
truth = [1,1,1,0]

num_samples, num_features = np.shape(input)
nn_iter = 1000
step_size = 0.3

n_nodes = [2]
n_hidden = 1
n_output = 1

X = input_setup(input)
weights = initialize_weights(X, n_nodes, n_hidden, n_output)

for i in range(nn_iter):
    
    for j in range(num_samples):

        xi = X[j,:]
        
        # feedforward
        h1 = sigmoid(weighted_sum(xi, weights[0]))
        h1 = np.hstack([1,h1])
        o1 = sigmoid(weighted_sum( h1, weights[1]))
        
        # backprop
        delta1 = (o1 - truth[j])*o1*(1-o1)
        delta2 =  delta1*hadamard(h1[1:]*(1-h1[1:]),weights[1][1:])
        rate1 = delta1*h1
        rate2 = np.matmul(np.array([xi]).T,np.array([delta2]))
        rate = [rate2,rate1]
        
        # grad descent
        weights[0] -= step_size*rate[0]
        weights[1] = sub(weights[1],step_size*rate[1])
    
#test    
h1 = sigmoid(np.matmul(X, weights[0]))
h1 = np.hstack([np.ones([4,1]),h1])
predict = sigmoid(np.matmul( h1, weights[1]))
print(predict)

#Example with one hidden layer with 2 nodes and two outputs
# testing NAND and OR
input = [[1,0],[0,1],[0,0],[1,1]]
truth = [[1,1],[1,1],[1,0],[0,1]]
num_samples, num_features = np.shape(input)
nn_iter = 1000
step_size = 0.3

n_nodes = [5]
n_hidden = 1
n_output = 2

X = input_setup(input)
weights = initialize_weights(X, n_nodes, n_hidden, n_output)

for i in range(nn_iter):
    
    for j in range(num_samples):
        
        xi = X[j,:]
        
        # feedforward
        h1 = sigmoid(weighted_sum(xi, weights[0]))
        h1 = np.hstack([1,h1])
        o = sigmoid(weighted_sum( h1, weights[1]))
        
        # backprop
        delta1 = (o - truth[j])*o*(1-o)
        delta2 = 0
        for i in range(n_output):
            
            delta2 += delta1[i]*h1[1:]*(1-h1[1:])*weights[1][1:,i]
            
        rate1 = np.matmul(np.array([h1]).T, np.array([delta1]))
        rate2 = np.matmul(np.array([xi]).T,np.array([delta2]))
        rate = [rate2,rate1]
        
        # grad descent
        weights[0] -= step_size*rate[0]
        weights[1] = sub(weights[1],step_size*rate[1])
        
#test
h1 = sigmoid(np.matmul(X, weights[0]))
h1 = np.hstack([np.ones([4,1]),h1])
predict = sigmoid(np.matmul( h1, weights[1]))
print(predict)

# Example with two hidden layers with 2 nodes and one output
#NAND again
input = [[1,0],[0,1],[0,0],[1,1]]
truth = [1,1,1,0]

num_samples, num_features = np.shape(input)
nn_iter = 1000
step_size = 0.5

n_nodes = [2,2]
n_hidden = 2
n_output = 1

X = input_setup(input)
weights = initialize_weights(X, n_nodes, n_hidden, n_output)

for i in range(nn_iter):
    
    for j in range(num_samples):
        
        xi = X[j,:]
        
        # feedforward
        h1 = sigmoid(weighted_sum(xi, weights[0]))
        h1 = np.hstack([1,h1])
        h2 = sigmoid(weighted_sum(h1, weights[1]))
        h2 = np.hstack([1,h2])
        o1 = sigmoid(weighted_sum(h2, weights[2]))
        
        # backprop
        delta1 = (o1 - truth[j])*o1*(1-o1)
        delta2 = delta1*hadamard(h2[1:]*(1-h2[1:]),weights[2][1:])
        delta3 = delta2*np.matmul(h1[1:]*(1-h1[1:]), weights[1][1:])
        rate1 = np.matmul(np.array([h2]).T,np.array([delta1]))
        rate2 = np.matmul(np.array([h1]).T,np.array([delta2]))
        rate3 = np.matmul(np.array([xi]).T,np.array([delta3]))
        rate = [rate3, rate2, rate1]
        
        # grad descent
        weights[0] -= step_size*rate[0]
        weights[1] -= step_size*rate[1]
        weights[2] -= step_size*rate[2]
        
#test
h1 = sigmoid(np.matmul(X, weights[0]))
h1 = np.hstack([np.ones([4,1]),h1])
h2 = sigmoid(np.matmul(h1, weights[1]))
h2 = np.hstack([np.ones([4,1]),h2])
predict = sigmoid(np.matmul(h2, weights[2]))
print(predict)

# Example with two hidden layers with 2 nodes and two outputs
# NAND and OR
input = [[1,0],[0,1],[0,0],[1,1]]
truth = [[1,1],[1,1],[1,0],[0,1]]
num_samples, num_features = np.shape(input)
nn_iter = 5000
step_size = 0.5

n_nodes = [2,2]
n_hidden = 2
n_output = 2

X = input_setup(input)
weights = initialize_weights(X, n_nodes, n_hidden, n_output)

for i in range(nn_iter):
    
    for j in range(num_samples):
        
        xi = X[j,:]
        
        # feedforward
        h1 = sigmoid(weighted_sum(xi, weights[0]))
        h1 = np.hstack([1,h1])
        h2 = sigmoid(weighted_sum(h1, weights[1]))
        h2 = np.hstack([1,h2])
        o = sigmoid(weighted_sum(h2, weights[2]))
        
        # backprop
        delta1 = (o - truth[j])*o*(1-o)
        delta2 = 0
        for i in range(n_output):
            
            delta2 += delta1[i]*h2[1:]*(1-h2[1:])*weights[2][1:,i]
            
        delta3 = 0
        for j in range(n_nodes[1]):
            
            delta3 += delta2[j]*h1[1:]*(1-h1[1:])*weights[1][1:,j]
            
        rate1 = np.matmul(np.array([h2]).T, np.array([delta1]))
        rate2 = np.matmul(np.array([h1]).T,np.array([delta2]))
        rate3 = np.matmul(np.array([xi]).T,np.array([delta3]))
        rate = [rate3,rate2,rate1]
        
        # grad descent
        weights[0] -= step_size*rate[0]
        weights[1] -= step_size*rate[1]
        weights[2] -= step_size*rate[2]
    
#test
h1 = sigmoid(np.matmul(X, weights[0]))
h1 = np.hstack([np.ones([4,1]),h1])
h2 = sigmoid(np.matmul(h1, weights[1]))
h2 = np.hstack([np.ones([4,1]),h2])
predict = sigmoid(np.matmul(h2, weights[2]))
print(predict)
