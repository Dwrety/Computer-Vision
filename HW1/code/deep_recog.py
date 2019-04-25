import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.distance import cdist


def build_recognition_system(vgg16, num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''
	train_file = np.load("../data/train_data.npz")
	train_data_path, labels = train_file['image_names'], train_file['labels']
	num_examples = len(labels)
	features = []
	for i in range(num_examples):
		print(i)
		feat = get_image_feature([i, train_data_path, vgg16, False])
		features.append(feat)
	# args = [[[i, train_data_path, vgg16, True]]for i in range(num_examples)]

	# with multiprocessing.Pool(processes=num_workers) as p:
	# 	p.starmap(get_image_feature, args)
	
	# features = []
	# for file in os.listdir('../temp/deep_features/'):
	# 	temp = np.load('../temp/deep_features/' + file)
	# 	features.append(temp)
	features = np.asarray(features)
	np.savez('trained_system_deep1.npz', features=features, labels=labels)
	return None


def evaluate_recognition_system(vgg16, num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	
	test_data = np.load("../data/test_data.npz")
	test_data_path, test_labels = test_data['image_names'], test_data['labels']
	trained_system = np.load("trained_system_deep1.npz")
	train_labels, train_features = trained_system['labels'], trained_system['features']
	
	i = 0
	labels_hat = []
	for test_file in test_data_path:
		print(i, test_file[0])
		feature = get_image_feature([i, test_data_path, vgg16, False])
		dist = distance_to_set(feature, train_features)
		classifer = np.argmin(dist)
		labels_hat.append(train_labels[classifer])
		i += 1

	labels_hat = np.asarray(labels_hat, dtype=int)
	np.save("labels_hat_deep1.npy", labels_hat)
	conf = np.zeros((8,8))
	i = 0
	for test in test_labels:
		conf[test, labels_hat[i]] += 1
		i += 1
	accuracy = np.trace(conf)/conf.sum()
	np.savez("deep_results1.npz", confusion_matrix=conf, accuracy=accuracy)
	print(conf)
	print('accuracy', accuracy)
	return conf, accuracy


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (1,3,H,W)
	'''
	# ----- TODO -----
	image = skimage.transform.resize(image, (224,224,3))
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]
	for i in range(3):
		image[:,:,i] = (image[:,:,i] - MEAN[i]) / STD[i]
	image = np.asarray([[image[:,:,i] for i in range(3)]]).astype(np.float32)
	image_processed = torch.from_numpy(image)
	return image_processed


def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''
	i, image_path, vgg16, if_save = args
	vgg16.classifier = nn.Sequential(*[vgg16.classifier[i] for i in range(5)])
	image_path = os.path.join('../data/', image_path[i][0])
	image = imageio.imread(image_path)
	if image.ndim == 2:
		image = np.dstack((image,image,image))
	if image.shape[2] == 4:
		image = np.delete(image, -1, axis=2)
	image_tensor = preprocess_image(image)
	conv = vgg16.features(image_tensor).flatten()
	feat = vgg16.classifier(conv).detach().numpy()

	if if_save:
		np.save('../temp/deep_features/{}.npy'.format(i), feat)
		print("Processing image", i)

	return feat


def distance_to_set(feature, train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	feature = feature.reshape(1, len(feature))
	dist = cdist(feature, train_features)[0]
	return dist


if __name__ == '__main__':

	deep_results = np.load("deep_results1.npz")
	conf, accuracy = deep_results["confusion_matrix"], deep_results["accuracy"]
	print(conf)
	print('accuracy', accuracy)