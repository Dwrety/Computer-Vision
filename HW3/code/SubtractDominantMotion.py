import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage.filters import median_filter, maximum_filter
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects
import numpy.linalg as nlg
from scipy.interpolate import RectBivariateSpline 


def SubtractDominantMotion(image1, image2, func=LucasKanadeAffine):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here

    threshold = 0.1
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    structure = np.array([[0,0,0,1,0,0,0],
    					  [0,1,1,1,1,1,0],
    					  [0,1,1,1,1,1,0],
    					  [1,1,1,1,1,1,1],
    					  [0,1,1,1,1,1,0],
    					  [0,1,1,1,1,1,0],
    					  [0,0,0,1,0,0,0]])
    H, W = image1.shape
    x = np.arange(W)
    y = np.arange(H)
    xv, yv = np.meshgrid(x, y)
    X = xv.flatten()
    Y = yv.flatten()
    interpSpline = RectBivariateSpline(y, x, image2)
    coordinate = np.stack([X, Y, np.ones(int(H*W))], axis=0)

    mask = np.zeros(image1.shape, dtype=bool)
    M = func(image1, image2)
    M_HOMO = np.append(M, [[0,0,1]], axis=0)
    coordinate_warp = M_HOMO.dot(coordinate)
    coordinate_x = coordinate_warp[0,:].copy()
    coordinate_y = coordinate_warp[1,:].copy()
    image2_warp = interpSpline(coordinate_y, coordinate_x, grid=False).reshape(H,W)
    Difference = abs(image1 - image2_warp)
    Difference = median_filter(Difference,size=3)

    mask[Difference>=threshold] = True
    mask = mask ^ remove_small_objects(mask, min_size=370, connectivity=1)
    mask = binary_dilation(mask,structure=structure)
    # mask = remove_small_objects(mask, min_size=150,connectivity=3)
    mask = binary_erosion(mask)
    return mask
