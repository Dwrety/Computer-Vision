import numpy as np
import scipy.ndimage as ndi 
from scipy.ndimage import affine_transform
from scipy.interpolate import RectBivariateSpline 
import numpy.linalg as nlg
import cv2


def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	H, W = It1.shape
	threshold = 1e-2
	p = np.zeros(6)
	delta_p = np.ones(6)
	num_iteration = 200


	x = np.arange(W)
	y = np.arange(H)
	xv, yv = np.meshgrid(x, y)
	X = xv.flatten()
	Y = yv.flatten()

	image1_interp = RectBivariateSpline(y,x,It)
	image2_interp = RectBivariateSpline(y,x,It1)

	coordinate = np.stack([X, Y, np.ones(int(H*W))], axis=0)

	epoch = 1
	while (nlg.norm(delta_p)> threshold) and (epoch < num_iteration):
		M_HOMO = np.append(M, [[0,0,1]], axis=0)

		coordinate_warp = M_HOMO.dot(coordinate)
		coordinate_x = coordinate_warp[0,:].copy()
		coordinate_y = coordinate_warp[1,:].copy()
		mask = np.logical_and(np.logical_and(coordinate_x>=0, coordinate_x<=W-1), np.logical_and(coordinate_y>=0, coordinate_y<=H-1))

		coordinate_x = coordinate_x[mask]
		coordinate_y = coordinate_y[mask]
		X_ = X[mask]
		Y_ = Y[mask]
		It_tmp = image1_interp(Y_, X_, grid=False)
		warp_frame = image2_interp(coordinate_y, coordinate_x, grid=False)
		error = It_tmp - warp_frame

		gradient_x = image2_interp(coordinate_y, coordinate_x,dx=0,dy=1,grid=False)
		gradient_y = image2_interp(coordinate_y, coordinate_x,dx=1,dy=0,grid=False)
		SDQ = np.stack([gradient_x*coordinate_x,
						gradient_x*coordinate_y,
						gradient_x,
						gradient_y*coordinate_x,
						gradient_y*coordinate_y,
						gradient_y], axis=1)

		Hessian = SDQ.T.dot(SDQ)
		b = SDQ.T.dot(error)
		delta_p = nlg.inv(Hessian).dot(b)
		p = p + delta_p
		M = np.array([[1+p[0], p[1], p[2]],
					  [p[3], 1+p[4], p[5]]])
		epoch += 1  

	return M
		
