import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 12
learning_rate = 5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here

initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')
if True:
    W_init = params['Wlayer1']
    W_init = np.reshape(W_init,(32,32,hidden_size))
    fig = plt.figure()
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(8, 8),
                 axes_pad=0.1,
                 )
    for i in range(hidden_size):
        grid[i].imshow(W_init[:,:,i],cmap='gray')
    plt.axis('off')
    plt.show()

train_loss = []
train_accuracy = []
valid_accuracy = []
valid_loss_plt = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        hidden1 = forward(xb,params,'layer1')
        probs = forward(hidden1,params,'output',softmax)
        # training loop can be exactly the same as q2!
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        # be sure to add loss and accuracy to epoch totals 
        avg_acc += acc 
        # backward
        delta1 = probs
        y_idxb = np.where(yb==1)[1]
        delta1[np.arange(probs.shape[0]),y_idxb] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['boutput'] -= learning_rate * params['grad_boutput']

    avg_acc /= batch_num
    total_loss = total_loss/train_x.shape[0]
    train_loss = np.append(train_loss,total_loss)
    train_accuracy = np.append(train_accuracy,avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
    # run on validation set and report accuracy! should be above 75%
    valid_hidden = forward(valid_x,params,'layer1')
    valid_probs = forward(valid_hidden,params,'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)
    valid_loss = valid_loss/valid_x.shape[0]
    valid_loss_plt = np.append(valid_loss_plt, valid_loss)
    valid_accuracy = np.append(valid_accuracy, valid_acc)

plt.figure()
plt.plot(np.arange(max_iters),train_loss)
plt.plot(np.arange(max_iters),valid_loss_plt)
plt.legend(['Training data','Validation data'])
plt.title('Loss averaged over the data, lr=0.005')
plt.show()

plt.figure()
plt.plot(np.arange(max_iters),train_accuracy)
plt.plot(np.arange(max_iters),valid_accuracy)
plt.legend(['Training data','Validation data'])
plt.title('Accuracy, lr=0.005')
plt.show()

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

W = params['Wlayer1']
W = np.reshape(W,(32,32,hidden_size))
fig = plt.figure()
grid = ImageGrid(fig, 111,
                 nrows_ncols=(8,8),
                 axes_pad=0.1,
                 )
for i in range(hidden_size):
    grid[i].imshow(W[:,:,i],cmap='gray')   
plt.axis('off') 
plt.show()

# # Q3.1.4

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
test_hidden = forward(test_x,params,'layer1')
test_probs = forward(test_hidden,params,'output',softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, test_probs)
idx_y = np.argmax(test_y,axis=1)
idx_probs = np.argmax(test_probs,axis=1)

for i in range(idx_y.shape[0]):
    confusion_matrix[idx_probs[i],idx_y[i]] += 1

plt.figure()
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()