import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt 


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    outsize = (1700, im1.shape[0])
    warp_2 = cv2.warpPerspective(im2, H2to1, outsize)

    pano_im = warp_2
    H, W = im1.shape[:2]
    pano_im[:H, :W,:] = im1
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    ## plain hard code stuff
    fixed_width = 1500
    H, W = im2.shape[:2]
    corner_points = np.zeros((4,3))
    corner_points[0,:] = [0,0,1]
    corner_points[1,:] = [W-1,0,1]
    corner_points[2,:] = [0,H-1,1]
    corner_points[3,:] = [W-1,H-1,1]
    corner_points = corner_points.T

    correspond_points = np.matmul(H2to1, corner_points)
    correspond_points = correspond_points/np.tile(correspond_points[2,:],(3,1))
    correspond_points = correspond_points[0:2,:]

    M = np.zeros((3,3))
    max_width = max(correspond_points[0,1], correspond_points[0,3])
    ratio = fixed_width / max_width
    ratio = 1
    M[0,0] = ratio - 0.1
    M[1,1] = ratio - 0.1
    M[2,2] = 1
    M[2,0] = 0
    max_height = min(correspond_points[1,0], correspond_points[1,1])
    M[2,1] = abs(max_height)
    M = M.T
    fixed_height = abs(max_height) + max(correspond_points[1,2], correspond_points[1,3])
    outsize = (fixed_width, int(round(fixed_height)))

    im1_warp = cv2.warpPerspective(im1, M, outsize);
    im2_warp = cv2.warpPerspective(im2, M.dot(H2to1), outsize);

    mask1 = np.zeros((im1.shape[0], im1.shape[1]))
    mask1[0,:] = 1
    mask1[-1,:] = 1
    mask1[:,0] = 1
    mask1[:,-1] = 1
    mask1 = distance_transform_edt(1-mask1)
    mask1 = mask1/mask1.max()
    mask1[np.isnan(mask1)] = 0
    mask_warp1 = cv2.warpPerspective(mask1, M, outsize)

    mask2 = np.zeros((im2.shape[0], im2.shape[1]))
    mask2[0,:] = 1
    mask2[-1,:] = 1
    mask2[:,0] = 1
    mask2[:,-1] = 1
    mask2 = distance_transform_edt(1-mask2)
    mask2 = mask2/mask2.max()
    mask2[np.isnan(mask2)] = 0
    mask_warp2 = cv2.warpPerspective(mask2, M.dot(H2to1), outsize)

    mask_to1 = mask_warp1/(mask_warp1+mask_warp2)
    mask_to2 = mask_warp2/(mask_warp1+mask_warp2)

    mask_to1[np.isnan(mask_to1)] = 1
    mask_to2[np.isnan(mask_to2)] = 1

    mask_to1 = np.dstack([mask_to1]*3)
    mask_to2 = np.dstack([mask_to2]*3)

    im1_warp = im1_warp /255
    im2_warp = im2_warp/ 255
    pano_im = np.multiply(mask_to1,im1_warp) + np.multiply(mask_to2,im2_warp)

    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # # plotMatches(im1,im2,matches,locs1,locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save("../results/q6_1.npy",H2to1)

    H2to1 = np.load("../results/q6_1.npy")

    # pano_im = imageStitching(im1, im2, H2to1)
    # cv2.imwrite('../results/6_1.jpg', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    pano_im = (pano_im*255).astype(np.uint8)
    cv2.imwrite('../results/q6_2.jpg', pano_im)
    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()