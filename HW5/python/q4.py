import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt 

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    # plt.imshow(image)
    image = skimage.restoration.denoise_wavelet(image,multichannel=True, convert2ycbcr=True)
    image = skimage.color.rgb2grey(image)
    thresh = skimage.filters.threshold_otsu(image)
    binary = image > thresh
    binary = skimage.morphology.binary_erosion(binary)
    # plt.imshow(binary,cmap='gray')
    bw = skimage.morphology.opening(binary)
    # binary = skimage.morphology.binary_erosion(binary)
    image_labels = skimage.measure.label(bw,connectivity=2,background=1)
    num_item = image_labels.max()
    bboxes = np.zeros((num_item,4))
    for i in range(num_item):
    	idx = np.where(image_labels==i+1)
    	y1 = idx[0].min()-20
    	y2= idx[0].max()+20
    	x1 = idx[1].min()-20
    	x2 = idx[1].max()+20
    	bboxes[i] = [y1,x1,y2,x2]
    bboxes_mean = np.mean(bboxes[:,2] - bboxes[:,0])
    idx_del = np.where(bboxes[:,2] - bboxes[:,0] < 0.65*bboxes_mean)
    bboxes = np.delete(bboxes,idx_del,axis = 0)
    return bboxes, bw

if __name__ == "__main__":
	image = skimage.io.imread('../images/03_haiku.jpg')
	findLetters(image)