import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA

train_mean = np.mean(train_x,axis=0)
train_shift = train_x - train_mean
cov_x = np.dot(train_shift.T,shift_x)

# gram trick 
u,sigma,vt = np.linalg.svd(cov_x)



# # rebuild a low-rank version
# lrank = None

# # rebuild it
# recon = None

# for i in range(5):
#     plt.subplot(2,1,1)
#     plt.imshow(train_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon[i].reshape(32,32).T)
#     plt.show()

# # build valid dataset
# recon_valid = None

# total = []
# for pred,gt in zip(recon_valid,valid_x):
#     total.append(psnr(gt,pred))
# print(np.array(total).mean())