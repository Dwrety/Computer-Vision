import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import numpy as np 
import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    '''
    #############################
    # TO DO ...
    # Generate testpattern here
    compareX, compareY = [], []
    for i in range(nbits):

        ## uniform distribution
        # X = np.random.randint(-patch_width//2, patch_width//2+1)
        # Y = np.random.randint(-patch_width//2, patch_width//2+1)

        ## normal distribution
        X = np.clip(np.around(np.random.normal(0, patch_width/5)), -4,4)
        Y = np.clip(np.around(np.random.normal(0, patch_width/5)), -4,4) 


        compareX.append(int(X*patch_width + Y + patch_width**2//2))

        # X = np.random.randint(-patch_width//2, patch_width//2+1)
        # Y = np.random.randint(-patch_width//2, patch_width//2+1)

        X = np.clip(np.around(np.random.normal(0, patch_width/5)), -4,4)
        Y = np.clip(np.around(np.random.normal(0, patch_width/5)), -4,4) 

        compareY.append(int(X*patch_width + Y + patch_width**2//2))
    compareX = np.clip(compareX,0,80).astype(int)
    compareY = np.clip(compareY,0,80).astype(int)

    return  compareX, compareY


def computeBrief(im, compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    H, W = im.shape[0], im.shape[1]
    locsDoG, gauss_pyramid = DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12)
    # print(len(locsDoG))
    locs = []
    desc = []
    for index in locsDoG:
        if index[0] >= 4 and index[0] <= W-5:
            if index[1] >= 4 and index[1] <= H-5:
                locs.append(index)
                vec = []
                for i in range(len(compareX)):
                    X_1 = compareX[i]//9
                    Y_1 = compareX[i]-X_1*9
                    X_2 = compareY[i]//9
                    Y_2 = compareY[i]-X_2*9
                    X_1 -= 4  
                    Y_1 -= 4  
                    X_2 -= 4  
                    Y_2 -= 4  
                    index_1 = (index[1]+Y_1, index[0]+X_1)
                    index_2 = (index[1]+Y_2, index[0]+X_2)
                    phi = int(im[index_1[0], index_1[1]]>im[index_2[0], index_2[1]])
                    vec.append(phi)
                vec = np.asarray(vec)
                desc.append(vec)
    desc = np.asarray(desc)
    locs = np.asarray(locs)
    return locs, desc


def briefLite(im, verbose=False):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    test_pattern_file = '../results/testPattern.npy'
    if os.path.isfile(test_pattern_file):
        if verbose:
            print("Reading test patches")
        # load from file if exists
        compareX, compareY = np.load(test_pattern_file)

    else:
        # produce and save patterns if not exist
        # print("Generate random patches")
        compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        if verbose:
            print("Cannot find directory: /results, creating one!")
        os.mkdir('../results')
    if not os.path.isfile(test_pattern_file):
        if verbose:
            print("Saving test patches")
        np.save(test_pattern_file, [compareX, compareY])
   
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    locs, desc = computeBrief(im, compareX, compareY)
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    imd = '_pf_scan_scaled_to_pf_desk'
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.savefig('../results/Q_2_4{}.jpg'.format(imd))
    plt.show()
    

if __name__ == '__main__':

    ## test makeTestPattern
    compareX, compareY = makeTestPattern()

    ## load test pattern for Brief
    # test_pattern_file = '../results/testPattern.npy'

    ## test briefLite
    # im = cv2.imread('../data/model_chickenbroth.jpg')
    # locs, desc = briefLite(im)  
    # fig = plt.figure()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    # plt.plot(locs[:,0], locs[:,1], 'r.')
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)

    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)

    # im1 = cv2.imread('../data/incline_L.png')
    # im2 = cv2.imread('../data/incline_R.png')
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)

    # im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    # im2 = cv2.imread('../data/pf_desk.jpg')
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)

    # pf_desk.jpg # pf_floor.jpg # pf_floor_rot.jpg # pf_pile.jpg # pf_stand.jpg