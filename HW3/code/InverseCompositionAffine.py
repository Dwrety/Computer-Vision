import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage as ndi 
from scipy.ndimage import affine_transform
import numpy.linalg as nlg
from scipy.interpolate import interp2d


def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	H, W = It1.shape
	threshold = 0.1
	delta_p = np.ones(6)
	num_iteration = 200
	epoch = 1 

	x = np.arange(W)
	y = np.arange(H)
	xv, yv = np.meshgrid(x, y)
	X = xv.flatten()
	Y = yv.flatten()

	image1_interp = RectBivariateSpline(y,x,It)
	image2_interp = RectBivariateSpline(y,x,It1)

	dx_vector = image1_interp(Y, X, dx=0, dy=1, grid=False)
	dy_vector = image1_interp(Y, X, dx=1, dy=0, grid=False)

	SDQ = np.stack([dx_vector*X,
					dx_vector*Y,
					dx_vector,
					dy_vector*X,
					dy_vector*Y,
					dy_vector], axis=1)
	Hessian = SDQ.T.dot(SDQ)
	coordinate = np.stack([X, Y, np.ones(int(H*W))], axis=0)

	while (nlg.norm(delta_p)> threshold) and (epoch <= num_iteration):
		M_HOMO = np.append(M, [[0,0,1]], axis=0)

		coordinate_warp = M_HOMO.dot(coordinate)
		coordinate_x = coordinate_warp[0,:].copy()
		coordinate_y = coordinate_warp[1,:].copy()
		warp_frame = image2_interp(coordinate_y, coordinate_x, grid=False)
		It_tmp = image1_interp(Y, X, grid=False)
		error = warp_frame - It_tmp
		b = SDQ.T.dot(error)
		delta_p = nlg.inv(Hessian).dot(b)
		dp_mat = np.array([[1+delta_p[0], delta_p[1],delta_p[2]],[delta_p[3],1+delta_p[4],delta_p[5]],[0,0,1]])
		M = M.dot(nlg.inv(dp_mat))
		epoch += 1

	return M
