import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]

def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    r = np.sqrt(6/(in_size+out_size))
    W = np.random.uniform(-r,r,size=(in_size, out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = X.dot(W)+b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)
    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    m = x.max(1).reshape(x.shape[0],1)
    x = x-m
    exp = np.exp(x)
    den = exp.sum(1)
    res = exp/den[:,None]
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    loss = -np.sum(y*np.log(probs))
    idx = probs.argmax(axis=1)
    predict = (idx[:,None] == np.arange(probs.shape[1])).astype(float)
    acc = np.sum(predict*y)/y.shape[0]
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    # I = activation_deriv(post_act)
    delta = delta*activation_deriv(post_act)
    # grad_W = np.zeros(W.shape)
    # grad_b = np.zeros(b.shape)
    grad_X = delta.dot(W.T)
    # for i in range(delta.shape[0]):
        # grad_W = grad_W+np.outer(X[i],delta[i])
        # grad_b = grad_b+delta[i]
    # grad_W = grad_W/delta.shape[0]
    # grad_b = grad_b/delta.shape[0]
    grad_W = X.T.dot(delta)
    grad_b = np.sum(delta, axis=0)
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    num_sample = x.shape[0]
    idx = np.random.permutation(np.arange(num_sample))
    idx = np.split(idx, num_sample/batch_size)
    for i in range(len(idx)):
        batches.append([x[idx[i]],y[idx[i]]])
    return batches
