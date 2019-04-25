import numpy as np
import scipy.io
from nn import *
from collections import Counter
from util import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

def squared_error_loss(input_layer, output_layer):
    loss = np.sum((output_layer - input_layer)**2)
    return loss

# initialize layers here
input_size = train_x.shape[1]
initialize_weights(input_size,batch_size,params,'layer1')
initialize_weights(batch_size,batch_size,params,'layer2')
initialize_weights(batch_size,batch_size,params,'layer3')
initialize_weights(batch_size,input_size,params,'output')
assert(params['Wlayer1'].shape == (input_size,batch_size))
assert(params['Wlayer2'].shape == (batch_size,batch_size))
assert(params['Wlayer3'].shape == (batch_size,batch_size))
assert(params['blayer1'].shape == (batch_size,))
assert(params['blayer2'].shape == (batch_size,))
assert(params['blayer3'].shape == (batch_size,))

mw_Wlayer1 = 0
mw_Wlayer2 = 0
mw_Wlayer3 = 0
mw_Woutput = 0
mw_blayer1 = 0
mw_blayer2 = 0
mw_blayer3 = 0
mw_boutput = 0


# should look like your previous training loops
loss_plt = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'layer2',relu)
        h3 = forward(h2,params,'layer3',relu)
        output_layer = forward(h3,params,'output')
        loss = squared_error_loss(xb, output_layer)
        total_loss += loss

        delta1 = output_layer
        delta1 = 2*(output_layer - xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)

        mw_Wlayer1 = 0.9 * mw_Wlayer1 - learning_rate * params['grad_Wlayer1']
        params['Wlayer1'] += mw_Wlayer1
        
        mw_Wlayer2 = 0.9 * mw_Wlayer2 - learning_rate * params['grad_Wlayer2']
        params['Wlayer2'] += mw_Wlayer2
        
        mw_Wlayer3 = 0.9 * mw_Wlayer3 - learning_rate * params['grad_Wlayer3']
        params['Wlayer3'] += mw_Wlayer3
        
        mw_Woutput = 0.9 * mw_Woutput - learning_rate * params['grad_Woutput']
        params['Woutput'] += mw_Woutput
        
        mw_blayer1 = 0.9 * mw_blayer1 - learning_rate * params['grad_blayer1']
        params['blayer1'] += mw_blayer1
        
        mw_blayer2 = 0.9 * mw_blayer2 - learning_rate * params['grad_blayer2']
        params['blayer2'] += mw_blayer2
        
        mw_blayer3 = 0.9 * mw_blayer3 - learning_rate * params['grad_blayer3']
        params['blayer3'] += mw_blayer3
        
        mw_boutput = 0.9 * mw_boutput - learning_rate * params['grad_boutput']
        params['boutput'] += mw_boutput

        # params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        # params['Wlayer2'] -= learning_rate * params['grad_Wlayer2']
        # params['Wlayer3'] -= learning_rate * params['grad_Wlayer3']
        # params['Woutput'] -= learning_rate * params['grad_Woutput']
        # params['blayer1'] -= learning_rate * params['grad_blayer1']
        # params['blayer2'] -= learning_rate * params['grad_blayer2']
        # params['blayer3'] -= learning_rate * params['grad_blayer3']
        # params['boutput'] -= learning_rate * params['grad_boutput']
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
    total_loss = total_loss/train_x.shape[0]
    loss_plt = np.append(loss_plt,total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

plt.figure()
plt.plot(np.arange(max_iters),loss_plt)
plt.title('Loss averaged over the data')
plt.show()  

    
# visualize some results
# Q5.3.1
idx = np.array([0,1,100,101,200,201,300,301,400,401])
xb = valid_x[idx]
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr
# # evaluate PSNR
# # Q5.3.2

total_psnr = 0
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'output')
for i in range(valid_x.shape[0]):
    out_psnr = psnr(valid_x[i],out[i],data_range=None)
    total_psnr += out_psnr
avg_psnr = total_psnr/valid_x.shape[0]
print(avg_psnr)
