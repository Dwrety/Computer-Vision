import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import matplotlib.pyplot as plt
from visual_words import get_visual_words
import multiprocessing


def build_recognition_system(num_workers=1):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	train_file = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")
	train_data, labels = train_file['image_names'], train_file['labels']
	layer_num = 3
	features = []
	i = 0
	for file in train_data:
		print(i, file[0])
		features.append(get_image_feature(file, dictionary, 3))
		i += 1
	features = np.asarray(features)
	np.savez('trained_system.npz', dictionary=dictionary, features=features, labels=labels, layer_num=layer_num)
	# args = [[[train_data, i], dictionary, 3] for i in range(len(train_data))]
	# with multiprocessing.Pool(processes=num_workers) as p:
	# 	res = p.starmap(get_image_feature, args)
	# res = np.asarray(res)
	return None
	

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''
	test_data = np.load("../data/test_data.npz")
	test_data, test_labels = test_data['image_names'], test_data['labels']

	trained_system = np.load("trained_system.npz")
	train_labels, train_histograms, dictionary = trained_system['labels'], trained_system['features'], trained_system['dictionary']

	i = 0
	labels_hat = []
	for test_file in test_data:
		print(i, test_data[0])
		hist_all = get_image_feature(test_file, dictionary, 3)
		sim = distance_to_set(hist_all, train_histograms)
		labels_hat.append(train_labels[np.argmax(sim)])
		i += 1
	labels_hat = np.asarray(labels_hat, dtype=int)
	np.save('labels_hat.npy', labels_hat)
	conf = np.zeros((8,8))
	i = 0
	for test in test_labels:
		conf[test, labels_hat[i]] += 1
		i += 1

	accuracy = np.trace(conf)/conf.sum()
	print(conf)	
	print('accuracy', accuracy)
	return conf, accuracy


def get_image_feature(file_path, dictionary, layer_num):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	image_path = os.path.join('../data/', file_path[0])
	image = imageio.imread(image_path)
	wordmap = get_visual_words(image, dictionary)
	dict_size = len(dictionary)
	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size)
	return feature


def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	sim = np.minimum(word_hist, histograms).sum(axis=1)
	return sim


def get_feature_from_wordmap(wordmap, dict_size, norm=True):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	# ----- TODO -----
	hist, bin_edges = np.histogram(wordmap, bins=dict_size, range=(0, dict_size), density=norm)
	return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	# ----- TODO -----
	# layer_num = 3, i = 0, 1, 2
	L = layer_num - 1
	cell_h = int(wordmap.shape[0]/2**L) 
	cell_w = int(wordmap.shape[1]/2**L) 
	l2 = []
	for i in range(L**2):
		for j in range(L**2):
			res = get_feature_from_wordmap(wordmap[i*cell_h:i*cell_h+cell_h, j*cell_w:j*cell_w+cell_w], dict_size, norm=False)
			l2.append(res)
	l2 = np.asarray(l2)
	l1 = []	
	l1.append(l2[0]+l2[1]+l2[4]+l2[5])
	l1.append(l2[2]+l2[3]+l2[6]+l2[7])
	l1.append(l2[8]+l2[9]+l2[12]+l2[13])
	l1.append(l2[10]+l2[11]+l2[14]+l2[15])
	l1 = np.asarray(l1)
	l0 = l1[0]+l1[1]+l1[2]+l1[3]
	num_features = l0.sum()
	hist_all = np.concatenate((0.25*l0, 0.25*l1.flatten(), 0.5*l2.flatten()))/num_features
	return hist_all


if __name__ == '__main__':
	test_data = np.load("../data/test_data.npz")
	test_data, test_labels = test_data['image_names'], test_data['labels']

	labels_hat = np.load('labels_hat.npy')
	conf = np.zeros((8,8))
	i = 0
	for test in test_labels:
		conf[test, labels_hat[i]] += 1
		i += 1

	accuracy = np.trace(conf)/conf.sum()
	np.savez("results.npz", confusion_matrix=conf, accuracy=accuracy)
	print(conf)	
	print('accuracy', accuracy)
