from BRIEF import briefLite, makeTestPattern, briefMatch, plotMatches
import cv2
import matplotlib.pyplot as plt 
import os 
import numpy as np


def rotation(im):
	rows,cols = im.shape[:2]
	theta = []
	num_match = []
	for i in range(37):
		theta.append(i*10)
		rot = cv2.getRotationMatrix2D((cols/2,rows/2),i*10,1)
		im_ = res = cv2.warpAffine(im,rot,(cols,rows))
		locs1, desc1 = briefLite(im)
		locs2, desc2 = briefLite(im_)
		matches = briefMatch(desc1, desc2)
		num_match.append(len(matches))
	return theta, num_match


if __name__ =='__main__':
	im = cv2.imread('../data/model_chickenbroth.jpg')
	test_pattern_file = '../results/testPattern.npy'
	theta, num_match = rotation(im)		
	plt.bar(theta, num_match, width = 15)
	plt.xlim((0, 360))
	plt.xlabel('theta')
	plt.ylabel("Number of Matches")
	plt.title("Interest point matching -- Rotation")
	plt.savefig('../results/Q_2_5.jpg')
	plt.show()