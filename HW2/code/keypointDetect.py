import numpy as np
import cv2
import skimage
import scipy.ndimage as ndi
import matplotlib.pyplot as plt 


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    for i in range(len(levels)-1):
    	DoG_pyramid.append(gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    ##################
    hx = cv2.Sobel(DoG_pyramid, -1, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    hy = cv2.Sobel(DoG_pyramid, -1, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    hxx = cv2.Sobel(hx, -1, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    hyy = cv2.Sobel(hy, -1, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    hxy = cv2.Sobel(hx, -1, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    hyx = cv2.Sobel(hx, -1, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    trace = hxx + hyy
    determinant = hxx*hyy - hxy*hyx
    principal_curvature = trace**2/(determinant+1e-10)
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    layer_print = np.zeros((3,3), dtype=int)
    layer_print[1,1] = 1
    footprint = np.stack([layer_print, np.ones((3,3)) ,layer_print], axis=-1)
    mask_max = ndi.maximum_filter(DoG_pyramid, footprint=footprint, mode='nearest') == DoG_pyramid
    mask_min = ndi.minimum_filter(DoG_pyramid, footprint=footprint, mode='nearest') == DoG_pyramid
    maxima = DoG_pyramid * mask_max
    minima = DoG_pyramid * mask_min
    local_extreme = np.argwhere(((abs(maxima)>th_contrast)&(abs(principal_curvature)<th_r))
                                |((abs(minima)>th_contrast)&(abs(principal_curvature)<th_r))) 
    locsDoG = local_extreme[:,[1,0,2]]
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    gauss_pyramid = createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4])
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    # im_pyr = createGaussianPyramid(im)
    # # print(im_pyr[:,:,0])
    # # displayPyramid(im_pyr)
    # # test DoG pyramid
    # DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # # displayPyramid(DoG_pyr)
    # # test compute principal curvature
    # pc_curvature = computePrincipalCurvature(DoG_pyr)
    # # displayPyramid(pc_curvature)
    # # # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locsDoG[:,0], locsDoG[:,1], 'r.')
    plt.draw()
    plt.savefig('../results/Q_1.5.jpg')
    plt.show()


