import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from numpy.linalg import inv, svd, det
from planarH import computeH


def compute_extrinsics(K,H):

	H= inv(K).dot(H)
	U,S,Vt = svd(H[:,0:2],True)

	I = np.array([[1,0],[0,1],[0,0]])
	R = np.zeros((3,3))
	R_12 = U.dot(I)
	R_12 = R_12.dot(Vt)
	R_3 = np.cross(R_12[:,0], R_12[:,1])
	R[:,0:2] = R_12
	R[:,-1] = R_3
	if det(R)<0:
		R[:,-1] = -1 * R[:,-1]
	lambda_prime = 0
	for i in range(3):
		for j in range(2):
			if not np.isnan(H[i,j]/R[i,j]):
				lambda_prime += H[i,j]/R[i,j]
	lambda_prime /= 6
	t = H[:,-1]/lambda_prime
	return R, t

def project_extrinsics(K, W, R, t):
	point_set = np.loadtxt("../data/sphere.txt")
	point_set = np.append(point_set, np.ones((1,point_set.shape[1])), axis=0)
	R_t = np.zeros((3,4))
	R_t[:,0:3] = R
	R_t[:,-1] = t
	warp_matrix = K.dot(R_t)
	warp_set = warp_matrix.dot(point_set)
	warp_set = warp_set/np.tile(warp_set[2,:],(3,1))
	warp_set = warp_set[0:2,:]

	t_ = np.ones(warp_set.shape)
	t_[0] = 313.
	t_[1] = 636.
	warp_set = warp_set + t_
	
	img = cv2.imread("../data/prince_book.jpeg")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	fig = plt.figure(1)
	plt.imshow(img)
	plt.plot(warp_set[0],warp_set[1],'.c',linewidth=1,markersize=1)
	plt.draw()
	plt.axis('off')
	plt.savefig("../results/Q_7.jpg",bbox_inches='tight')
	plt.show()
	return None 


if __name__ == '__main__':

	W = np.array([[0.0, 18.2, 18.2, 0.0],
				[0.0, 0.0, 26.0, 26.0],
				[0.0, 0.0, 0.0, 0.0]])

	X = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
	p1 = X
	p2 = W[0:2,:]
	H = computeH(p1, p2)

	K = np.array([[3043.72, 0.0, 1196.0],[0.0, 3043.72, 1604.0],[0.0, 0.0, 1.0]])
	R,t = compute_extrinsics(K, H)

	project_extrinsics(K, W, R, t)
