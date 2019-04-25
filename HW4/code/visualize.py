'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter3
'''

import numpy as np  
import submission as sub 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d, Axes3D
from findM2 import findM2
import helper


points = np.load("../data/templeCoords.npz")
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
F = np.load("../results/q2_1.npz")['F']
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
M1 = np.hstack([np.eye(3),np.zeros((3,1))])

X1 = points['x1']
Y1 = points['y1']
N = X1.shape[0]
X1 = X1.reshape(N)
Y1 = Y1.reshape(N)

X2, Y2 = np.zeros(N), np.zeros(N)
for i in range(N):
	X2[i], Y2[i] = sub.epipolarCorrespondence(im1, im2, F, X1[i], Y1[i])

pts1 = np.stack([X1, Y1], axis=1)
pts2 = np.stack([X2, Y2], axis=1)

M2, P = findM2(pts1, pts2, F, K1, K2)
C1=K1.dot(M1)
C2 = K2.dot(M2)
# np.savez('../results/q4_2.npz',F=F,M1=M1,M2=M2,C1=C1,C2=C2)
M2, P = sub.bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot(P[:,0], P[:,1], P[:,2],'bo')
ax.set_zlim(3.4,4.1)
ax.set_ylim(-0.6,0.6)
plt.show()