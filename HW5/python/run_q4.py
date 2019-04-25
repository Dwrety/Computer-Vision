import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def sort_classes(x):
    from sklearn.cluster import MeanShift
    x = x.reshape(-1,1)
    clf = MeanShift(bandwidth=100)
    clf.fit(x)
    return clf.labels_ 

import string
import pickle
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    bboxes[bboxes < 0] = 0
    line_of_text = []
    # print(bboxes)
    # print(bboxes)
    # plt.imshow(bw,cmap='gray')
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                             fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    classes = sort_classes(bboxes[:,2])
    indexes = np.unique(classes, return_index=True)[1]
    line = [classes[index] for index in sorted(indexes)]
    bboxes = np.c_[bboxes,classes].astype(int)

    for l in line:
        bboxes_line = bboxes[bboxes[:,4]==l]
        bboxes_line = bboxes_line[bboxes_line[:,1].argsort()]
        previous = bboxes_line[0,3].copy()
        for box in bboxes_line:
            y1,x1,y2,x2 = box[0:4]
            image = bw[y1:y2,x1:x2]
            # image[0:4,:] = 1
            # image[-5:,:] = 1
            # image[:,0:4] = 1
            # image[:,-5:] = 1
            image = skimage.transform.resize(image, (32,32),anti_aliasing=False).T
            x = image.reshape(1,1024)

            hidden = forward(x,params,'layer1')
            probs = forward(hidden,params,'output',softmax)
            character = letters[np.argmax(probs)]
            if x1-previous>=70:
                line_of_text.append(' ')
            line_of_text.append(character)

            previous = x2.copy()
        line_of_text.append('\n')
    f = open('output.txt', 'a') 
    for item in line_of_text:
        f.write("%s" % item)    
    
    # bboxes = np.c_[bboxes,np.zeros(bboxes.shape[0])]
    # bboxes = bboxes[bboxes[:,2].argsort()]
    # box_maxr = bboxes[:,2]
    # print(box_maxr)
    
    # box_class = box_maxr[0]
    # c = 0
    # for i in range(bboxes.shape[0]):
    #     if np.absolute(box_maxr[i] - box_class) < np.mean(bboxes[:,2]-bboxes[:,0])/2:
    #         bboxes[i,4] = c
    #         if box_maxr[i] > box_class:
    #             box_class = box_maxr[i]
    #     else:
    #         c += 1
    #         bboxes[i,4] = c
    #         box_class = box_maxr[i]

    
    # for i in range(c+1):
    #     box_class = bboxes[np.where(bboxes[:,4] == i),:][0]
    #     box_minc = box_class[:,1]
    #     box_class = box_class[box_minc.argsort()]
    #     bboxes[np.where(bboxes[:,4] == i),:] = box_class
    
    # # crop the bounding boxes
    # # note.. before you flatten, transpose the image (that's how the dataset is!)
    # # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # # load the weights
    # # run the crops through your neural network and print them out
    
    # import pickle
    # import string
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # params = pickle.load(open('q3_weights.pickle','rb'))
    
    # bboxes_resized = np.zeros((bboxes.shape[0],1024))
    # for i in range(bboxes.shape[0]):
    #     box = bw[int(bboxes[i][0]):int(bboxes[i][2]),int(bboxes[i][1]):int(bboxes[i][3])]
    #     box[0:19,:] = True
    #     box[-20:-1,:] = True
    #     box[:,0:19] = True
    #     box[:,-20:-1] = True
    #     if box.shape[0] <= box.shape[1]:
    #         n = box.shape[1] - box.shape[0]
    #         box = np.pad(box,((int(n/2),int(n/2)),(0,0)),'constant',constant_values = 1)
    #     else:
    #         n = box.shape[0] - box.shape[1]
    #         box = np.pad(box,((0,0),(int(n/2),int(n/2))),'constant',constant_values = 1)
    #     box = skimage.transform.resize(box, (32,32),anti_aliasing=False)
    #     box = box.T
    #     box = skimage.morphology.erosion(box)
    #     box = np.reshape(box,1024)
    #     bboxes_resized[i] = box

    # h1 = forward(bboxes_resized,params,'layer1')
    # probs = forward(h1,params,'output',softmax)
    # bboxes_recog = np.argmax(probs,1)
    
    # for i in range(c+1):
    #     output = bboxes_recog[bboxes[:,4] == i]
    #     output_l = output.shape[0]
    #     output_new = []
    #     minc = bboxes[bboxes[:,4] == i,1]
    #     maxc = bboxes[bboxes[:,4] == i,3]
    #     for j in range(output_l-1):
    #         if minc[j+1] - maxc[j] < np.mean(bboxes[:,3]-bboxes[:,1])/2:
    #             output_new = np.append(output_new,output[j])
    #         else:
    #             output_new = np.append(output_new,output[j])
    #             output_new = np.append(output_new,100)
    #     output_new = np.append(output_new,output[-1])
    #     output_new[output_new <= 25] += 65
    #     output_new[output_new <= 35] += 48
    #     output_new[output_new == 100] = 0
    #     output_new = list(output_new)
    #     for i in range(len(output_new)):
    #         output_new[i] = chr(int(output_new[i]))
    #     print(' '.join(output_new))

