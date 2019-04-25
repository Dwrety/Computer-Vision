'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, p1, p2, R and P to q3_3.npz
'''


import numpy as np 
import submission as sub 
import helper


def findM2(pts1, pts2, F, K1, K2):
	E = sub.essentialMatrix(F, K1, K2)
	M2s = helper.camera2(E)
	M1 = np.hstack([np.eye(3),np.zeros((3,1))])
	C1 = K1.dot(M1)
	for i in range(4):
		C2 = K2.dot(M2s[:,:,i])
		P, err = sub.triangulate(C1, pts1, C2, pts2)
		if P.min(0)[2] > 0:
			break
	print("Reprojection error is ", err)
	np.savez('../results/q3_3.npz', M2=M2s[:,:,i],C2=C2, P=P)
	return M2s[:,:,i], P