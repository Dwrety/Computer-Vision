import numpy as np
import scipy.ndimage
import os,time
from util import get_VGG16_weights
import torchvision
import imageio
import skimage.transform


def extract_deep_feature(x, vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	x = skimage.transform.resize(x, (224,224,3))
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]

	# SINGLE BATCH NORMALIZATION
	for i in range(3):
		x[:,:,i] = (x[:,:,i] - MEAN[i]) / STD[i]
	

	conv_block = 0
	for layer in vgg16_weights[:-2]: # UPTO FC7 LAYER
		if layer[0] == 'conv2d':
			x = multichannel_conv2d(x, layer[1], layer[2])
		if layer[0] == 'relu':
			x = relu(x)
		if layer[0] == 'linear':
			x = linear(x, layer[1], layer[2])
		if layer[0] == 'maxpool2d':
			x = max_pool2d(x, layer[1])
			conv_block += 1
			if conv_block == 5:
				x = x.flatten()
	return x			


def multichannel_conv2d(x, weight, bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''

	# convert to shape (input_dim, H, W)
	H, W, input_dim = x.shape
	output_dim = bias.shape[0]
	x = np.asarray([ x[:,:,i] for i in range(x.shape[2])]) # x ---> (input_dim, H, W)

	feat = np.zeros((output_dim, H, W))
	
	i = 0
	for j in weight:
		# feat[i] has dimension of (H, W)
		k = 0
		for channel in x:
			feat[i] += scipy.ndimage.convolve(channel, np.flipud(np.fliplr(j[k])), mode='nearest')
			k += 1
		feat[i]	+= bias[i]
		i += 1
	feat = np.dstack(feat)	
	return feat


def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x, 0)


def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	pad = False
	method = 'MaxPooling'
	H, W = x.shape[:2]

	_ceil = lambda x,y: int(np.ceil(x/float(y)))
	if pad:
		ny = _ceil(H, size)
		print(ny)
		nx = _ceil(W, size)
		print(nx)
		sz = (ny*size, nx*size) + x.shape[2]
		x_pad = np.full(sz, 0)
		x_pad[:H,:W] = x
	else:
		ny = H // size
		nx = W // size
		x_pad = x[:ny*size, :nx*size]	

	_shape = (ny, size, nx, size) + x.shape[2:]	

	if method == 'MaxPooling':
		res = np.nanmax(x_pad.reshape(_shape), axis=(1,3))
	else:
		res = np.nanmean(x_pad.reshape(_shape), axis=(1,3))	
	## out_H, out_W = int(np.floor(H / stride)), int(np.floor(W / stride))
	
	## # NO PADDING
	## # H_remainder, W_remainder = np.mod(H, stride), np.mod(W, stride)
	## y = np.zeros((out_H, out_W, D))
	
	## # FOR MAX POOLING
	## for i in range(out_H):
	## 	for j in range(out_W):
	## 		X = j * stride
	## 		Y = i * stride
	## 		pool = x[Y:Y+size,X:X+size]
	## 		y[i,j] = np.max(pool, axis=(0, 1))
	return res		
	



def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	return W.dot(x) + b


if __name__ == '__main__':
	vgg16_weights = get_VGG16_weights()
	image = imageio.imread('sun_adpubainqujzqhnh.jpg')
	feat = extract_deep_feature(image, vgg16_weights)
	print(feat.shape)