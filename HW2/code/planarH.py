import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
from numpy.linalg import eig, svd
import os 
import matplotlib.pyplot as plt


def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''


    # Reference Function in OpenCV
    H2to1, status = cv2.findHomography(p2.T, p1.T)

    # My Own Implementation of SVD
    # Denote source point as (u,v) and dest point as (x,y)
    if not p1.shape == p2.shape:
    	raise ValueError("Input points must have same dimensions.")
    num_points = p1.shape[1]
    A = np.zeros((2*num_points,9), dtype=float)
    for i in range(num_points):
    	ui, vi = p2[:,i]
    	xi, yi = p1[:,i]
    	A[i*2,3:6] = [-ui,-vi,-1.]
    	A[i*2,6:9] = [yi*ui,yi*vi,yi]
    	A[i*2+1,0:3] = [ui,vi,1.]
    	A[i*2+1,6:9] = [-xi*ui,-xi*vi,-xi]
    U,S,Vt = svd(A,True)
    H = (Vt[-1]/Vt[-1,-1]).reshape(3,3)
    H2to1 = H
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    bestH = None
    accuracy = []
    H_collection = []
    for i in range(num_iter):
    	random_4 = np.random.permutation(matches)
    	p1,p2 = [],[]
    	for j in range(4):
    		pt1 = locs1[random_4[j,0], :2]
    		pt2 = locs2[random_4[j,1], :2]
    		p1.append(pt1)
    		p2.append(pt2)
    	p1 = np.asarray(p1).T
    	p2 = np.asarray(p2).T
    	H = computeH(p1,p2)
    	H_collection.append(H)
    	n_inlier = 0.
    	for j in range(len(matches)):
    		pt1 = locs1[matches[j,0], :2]
    		pt2 = locs2[matches[j,1], :2]
    		n_inlier += compareAccuracy(pt1,pt2,H,tol)
    	accuracy.append(n_inlier)
    idx = np.argmax(accuracy)
    bestH = H_collection[idx]
    return bestH


def compareAccuracy(pt1, pt2, H, tol):
	pt2 = np.append(pt2,[1],axis=0)
	pt1 = np.append(pt1,[1],axis=0)
	warp = H.dot(pt2)
	warp = warp/(warp[2]+1e-12)
	return float(np.linalg.norm(pt1-warp)<=tol)


# useless function 1
def apply_homography(H,points):
    p = np.ones((len(points),3))
    p[:,:2] = points
    pp = np.dot(p,H.T)
    pp[:,:2]/=pp[:,2].reshape(len(p),1)
    return pp[:,:2]


if __name__ == '__main__':
    # im1 = cv2.imread('../data/incline_L.png')
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    rows,cols = im1.shape[:2]
    rot = cv2.getRotationMatrix2D((cols/2,rows/2),20,1)
    im2 = cv2.warpAffine(im1,rot,(cols,rows))
    # im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc2, desc1)
    bestH = ransacH(matches, locs2, locs1, num_iter=5000, tol=2)

    outsize = (im2.shape[1],im2.shape[0])
    im_out = cv2.warpPerspective(im1, bestH, outsize)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.imshow(im1)
    plt.title("Left Image")
    plt.subplot(132)
    plt.imshow(im2)
    plt.title("Right Image")
    plt.subplot(133)
    plt.imshow(im_out)
    plt.title("Warped Image")
    plt.show()

