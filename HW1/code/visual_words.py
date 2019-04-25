import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
from scipy.spatial.distance import cdist
import os,time
import matplotlib.pyplot as plt
import util
import random
from scipy.ndimage import gaussian_filter, gaussian_laplace


def filter_1(array, scale):
	return gaussian_filter(array, sigma=scale, output=np.float64, mode='nearest')

def filter_2(array, scale):
	return gaussian_laplace(array, sigma=scale, output=np.float64, mode='nearest')

def filter_3(array, scale):
	return gaussian_filter(array, sigma=scale, order=[1,0], output=np.float64, mode='nearest')

def filter_4(array, scale):
	return gaussian_filter(array, sigma=scale, order=[0,1], output=np.float64, mode='nearest')


def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	
	# ----- TODO -----
	if image.ndim == 2:
		image = np.dstack((image,image,image))
	if image.shape[2] == 4:
		image = np.delete(image, -1, axis=2)

	image = skimage.color.rgb2lab(image)
	l, a, b = image[:,:,0], image[:,:,1], image[:,:,2]
	filter_scales  = [1.0, 2.0, 4.0, 8.0, 8*np.sqrt(2)]
	result = []
	i = 0
	for scale in filter_scales:
		for f in [filter_1, filter_2, filter_4, filter_3]:

			for channel in [l, a, b]:
				filtered = f(channel, scale)
				result.append(filtered) 	
	rv = np.dstack(result)
	return rv


def get_visual_words(image, dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----
	filter_responses = extract_filter_responses(image).reshape(-1, 60)
	distance = cdist(filter_responses, dictionary, metric='euclidean')
	best_fit = np.argmin(distance, axis=1)
	return best_fit.reshape(image.shape[0], image.shape[1])


def compute_dictionary_one_image(i, alpha, train_data):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''
	
	image_path = os.path.join('../data/', train_data[i][0])
	save_path = '../temp/'
	image = imageio.imread(image_path)
	raw_responses = extract_filter_responses(image)
	shuffle = np.random.permutation(raw_responses.reshape(-1, raw_responses.shape[2]))
	np.save('../temp/200/{}.npy'.format(i), shuffle[:alpha])
	print('Finish computing dictionary one image of', i)
	return None


def compute_dictionary(num_workers=4, mode=1):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''
	K = 200
	train_file = np.load("../data/train_data.npz")
	train_data, train_labels = train_file['image_names'], train_file['labels']
	if mode == 1:
		args = [[i, 250, train_data] for i in range(len(train_data))]
		with multiprocessing.Pool(processes=num_workers) as p:
			p.starmap(compute_dictionary_one_image, args)
		return None
	if mode == 2:
		features = []
		for file in os.listdir('../temp/200/'):
			temp = np.load('../temp/200/' + file)
			features.append(temp)
		features = np.asarray(features)
		filter_responses = features.reshape(-1, 60)
		kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(filter_responses)
		dictionary = kmeans.cluster_centers_
		np.save('dictionary.npy', dictionary)
		return dictionary


if __name__ == "__main__":
	# compute_dictionary(num_workers=6, mode=1)
	# dictionary = compute_dictionary(num_workers=6, mode=2)
	# print(dictionary.shape)
	dictionary = np.load('dictionary.npy')
	train_file = np.load("../data/train_data.npz")
	train_data, train_labels = train_file['image_names'], train_file['labels']
	lst = np.random.permutation(train_data)[:5]
	i = 1
	for pic in lst:
		image = imageio.imread('../data/'+ pic[0])
		wordmap = get_visual_words(image, dictionary)
		util.save_wordmap(image ,wordmap, '{}.png'.format(i))
		i += 1