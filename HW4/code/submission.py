"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np 
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import helper
import sympy as sp
from sympy.matrices import Matrix
from sympy.solvers import solve
from findM2 import findM2
from scipy.ndimage import gaussian_filter
import cv2
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import axes3d, Axes3D


def eightpoint(pts1, pts2, M, REFINE=True):
    '''
    Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix 
    		specifically transform from pts1 to pts2
    '''
    # Replace pass by your implementation
    if pts1.shape != pts2.shape:
    	raise ValueError("The input arguments should have the same dimension! And please use cartesian coordinates")
    if pts1.shape[0] < 8:
    	raise ValueError("Input of at least eight points is required!")
    num_points = pts1.shape[0]
    # normalize all points
    p1 = pts1.copy()
    p2 = pts2.copy() 
    pts1 = pts1/M
    pts2 = pts2/M

    A = np.zeros((0, 9))
    for i in range(num_points):
    	a1 = pts2[i,0]*pts1[i,0]
    	a2 = pts2[i,0]*pts1[i,1]
    	a3 = pts2[i,0]
    	a4 = pts2[i,1]*pts1[i,0]
    	a5 = pts1[i,1]*pts1[i,1]
    	a6 = pts2[i,1]
    	a7 = pts1[i,0]
    	a8 = pts1[i,1]
    	a9 = 1.0
    	A = np.append(A, [[a1, a2, a3, a4, a5, a6, a7, a8, a9]], axis=0)
    U, L, VT = nlg.svd(A)
    F = VT[-1,:].reshape(3,3)

    # enforce rank 2 constraint and 
    if REFINE:
    	F = helper.refineF(F, pts1, pts2)
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(VT))
    T = np.diag([1/M, 1/M, 1.0])
    F = T.dot(F).dot(T)
    np.savez("../results/q2_1.npz", F=F, M=M)
    # F = F/F[2,2]
    return F


def sevenpoint(pts1, pts2, M):
    '''
    Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
    '''	
    # Replace pass by your implementation
    if pts1.shape != pts2.shape:
    	raise ValueError("The input arguments should have the same dimension! And please use cartesian coordinates")
    if pts1.shape[0] < 7 or pts1.shape[0] > 7:
    	raise ValueError("Input arguments should be seven cooresponding points pairs")
    F = None
    num_points = pts1.shape[0]
    # normalize all points
    p1 = pts1.copy()
    p2 = pts2.copy()
    T = np.diag([1/M, 1/M, 1.0])
    pts1 = 1/M*pts1
    pts2 = 1/M*pts2
    A = np.zeros((0, 9))
    for i in range(num_points):
    	a1 = pts2[i,0]*pts1[i,0]
    	a2 = pts2[i,0]*pts1[i,1]
    	a3 = pts2[i,0]
    	a4 = pts2[i,1]*pts1[i,0]
    	a5 = pts1[i,1]*pts1[i,1]
    	a6 = pts2[i,1]
    	a7 = pts1[i,0]
    	a8 = pts1[i,1]
    	a9 = 1.0
    	A = np.append(A, [[a1, a2, a3, a4, a5, a6, a7, a8, a9]], axis=0)
    U, L, VT = nlg.svd(A)
    F1 = VT[-1,:].reshape(3,3)
    F2 = VT[-2,:].reshape(3,3)

    # func = lambda alpha: nlg.det(alpha*F1+(1-alpha)*F2)
    # p3 = func(0)
    # p2 = (8*(func(1)-func(-1))-(func(2)-func(-2)))/12
    # p1 = 0.5*(func(1)+func(-1))-func(0)
    # p0 = (func(2) - func(-2) - 2*(func(1) - func(-1)))/12

    alpha = sp.Symbol('alpha')
    H = alpha*F1 + (1.0-alpha)*F2
    H = sp.Matrix(H)
    # print(H)
    Poly = H.det().as_poly().coeffs()
    # Poly = [p0, p1, p2, p3]
    lmb = np.roots(Poly)
    print(lmb)
    lmb = np.real(lmb)
    
    F_ = [l*F1+(1.0-l)*F2 for l in lmb if l>=0 and l<=1]
    F = []
    for f in F_:
    	f = helper.refineF(f, pts1, pts2)
    	u, s, vt = np.linalg.svd(f)
    	s[-1] = 0
    	f = u.dot(np.diag(s).dot(vt))
    	f = T.dot(f).dot(T)
    	F.append(f)
    F = np.array(F)
    if len(F) == 0:
    	raise Exception('Please pick another 7 pairs of points.')
    np.savez("../results/q2_2.npz", F=F, M=M)
    return F


def essentialMatrix(F, K1, K2):
    '''
    Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential Matrix
    '''
    # Replace pass by your implementation
    return K2.T.dot(F).dot(K1)


def triangulate(C1, pts1, C2, pts2):
    '''
    Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
    '''
    # Replace pass by your implementation
    N = pts1.shape[0]
    # pts1_Homo = np.column_stack((pts1, np.ones(N)))
    # pts2_Homo = np.column_stack((pts2, np.ones(N)))
    C11 = C1[0]
    C21 = C2[0]
    C12 = C1[1]
    C22 = C2[1]
    C13 = C1[2]
    C23 = C2[2]
    P = np.zeros((N,4))
    A = np.zeros((4,4))
    X1 = np.zeros((N,2))
    X2 = np.zeros((N,2))
    for i in range(N):
        A[0] = C11 - C13*pts1[i,0]
        A[1] = C12 - C13*pts1[i,1]
        A[2] = C21 - C23*pts2[i,0]
        A[3] = C22 - C23*pts2[i,1]
        U, S, VT = nlg.svd(A)
        del U, S
        P[i] = VT[-1]/VT[-1,-1]
        x1 = C1.dot(P[i])
        x2 = C2.dot(P[i])
        
        x1 = x1/x1[-1]
        x2 = x2/x2[-1]
        X1[i] = x1[0:2]
        X2[i] = x2[0:2]
    err = nlg.norm(pts1-X1)**2 + nlg.norm(pts2-X2)**2
    P = np.array(P)
    P = P[:,0:3]
    return P, err


def epipolarCorrespondence(im1, im2, F, x1, y1):
    '''
    Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
    '''
    # Replace pass by your implementation

    # im1_noise = np.random.random(im1.shape)
    # im2_noise = np.random.random(im2.shape)
    # factor = 1e-3
    # im1 = im1+im1_noise*factor
    # im2 = im2+im2_noise*factor
    X1_HOMO = np.array([x1, y1, 1.0])
    epipolarLine = F.dot(X1_HOMO)

    ly2 = np.arange(im2.shape[0], dtype=int)
    lx2 = np.rint(-1/epipolarLine[0]*(epipolarLine[2]+epipolarLine[1]*ly2)).astype(int)
    kernal_size = 13

    pad_size = kernal_size//2
    lx2 = lx2 + pad_size
    ly2 = ly2 + pad_size

    window = np.zeros((kernal_size, kernal_size))
    window[kernal_size//2, kernal_size//2] = 1
    kernal = gaussian_filter(window, sigma=6.5)
    kernal_3d = np.stack([kernal, kernal, kernal],axis=2)

    im1_pad = cv2.copyMakeBorder(im1, pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT)
    im2_pad = cv2.copyMakeBorder(im2, pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT)
    
    x1_pad, y1_pad = x1+pad_size, y1+pad_size
    window_1 = im1_pad[y1_pad-pad_size:y1_pad+pad_size+1,x1_pad-pad_size:x1_pad+pad_size+1,:]*kernal_3d

    match_idx = 0
    patch_dist = []
    location_dist = []

    for i in range(len(lx2)):
    	if ly2[i]-pad_size>=0 and ly2[i]-pad_size<=im2.shape[0]-1 and lx2[i]-pad_size>=0 and lx2[i]-pad_size<=im2.shape[1]-1:
    		# print((lx2[i], ly2[i]))
    		x2_pad, y2_pad = int(lx2[i]), int(ly2[i])
    		window_2 = im2_pad[y2_pad-pad_size:y2_pad+pad_size+1,x2_pad-pad_size:x2_pad+pad_size+1,:]*kernal_3d
    		norm = nlg.norm(window_1-window_2)
    		patch_dist.append(norm)

    patch_dist = np.array(patch_dist)
    patch_sort = np.argsort(patch_dist)
    for i in range(patch_dist.shape[0]):
    	match_idx = patch_sort[i]
    	x2 = lx2[match_idx]-pad_size
    	y2 = ly2[match_idx]-pad_size
    	X2_HOMO = np.array([x2, y2, 1.0])
    	l = nlg.norm(X1_HOMO-X2_HOMO)
    	if l <= 38:
    		break
    return x2, y2 


def ransacF(pts1, pts2, M):
    '''
    Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
    '''

    # Replace pass by your implementation
    func = eightpoint
    num_points = 24
    num_iterations = 10000
    inliers = []
    F_list = []
    epoch = 1
    N = pts1.shape[0]
    threshold = 1e-3
    lst = np.arange(N)
    p1 = np.column_stack((pts1, np.ones(N)))
    p2 = np.column_stack((pts2, np.ones(N)))
    print("Running RANSAC Algorithm, I suppressed the display of optimization so it won't get messy. \ncout every 1000 iterations")
    ckpt = 0
    while epoch <= num_iterations:
    	if (epoch - 1000*(epoch//1000)) == 0 :
    		ckpt += 1
    		print("Current iteration is {}".format(ckpt*1000))	
    	random = np.random.permutation(lst)[0:num_points]

    	F = func(pts1[random], pts2[random], M, REFINE=False)
    	F_list.append(F)
    	inlier = []
    	for i in range(N):
    		qury = F.dot(p1[i])
    		qury = p2[i].dot(qury)
    		if abs(qury) < threshold:
    			inlier.append(1)
    		else:
    			inlier.append(0)
    	inliers.append(inlier)
    	epoch += 1
    D = np.sum(inliers, axis=1)
    idx = np.argmax(D)
    F = F_list[idx]
    inlier = inliers[idx]
    inlier = np.array(inlier, dtype=bool)
    print("Inliers proportion: ", sum(inlier)/N)
    pts1 = pts1[inlier]
    pts2 = pts2[inlier]
    F = eightpoint(pts1, pts2, M)
    return F, inlier


def rodrigues(r):
    '''
    Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
    '''
    # Replace pass by your implementation
    R = np.eye(3)
    theta = nlg.norm(r)
    if theta == 0:
    	R = np.eye(3)
    else:
    	skew = np.array([[0,-r[2], r[1]],[r[2], 0, -r[0]],[-r[1], r[0], 0]])/theta
    	R = np.eye(3) + skew*np.sin(theta) + skew.dot(skew)*(1-np.cos(theta))
    return R


def invRodrigues(R): 
    '''
    Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
    '''
    # Replace pass by your implementation
    A = (R-R.T)/2
    rho = np.array([A[2,1], A[0,2], A[1,0]])
    s = np.linalg.norm(rho)
    c = (np.trace(R)-1)/2
    if abs(s)<1e-12 and abs(c-1)< 1e-12:
        r = np.array([0,0,0.])
    elif abs(s)<1e-12 and abs(c+1)<1e-12:
        d = np.eye(3)+R
        idx = np.argmax(np.diag(d))
        v = d[:,idx]
        u = v/np.linalg.norm(v)
        r = np.pi*u
        if abs(np.linalg.norm(r)-np.pi)<1e-12 and \
        ((abs(r[0])<1e-12 and abs(r[1])<1e-12 and r[2]<0) or (abs(r[0])<1e-12 and r[1]<0) or (r[0]<0)):
            r = -r
    else:
        theta = 0
        if c>0:
            theta = np.arctan(s/c)
        elif c<0:
            theta = np.pi + np.arctan(s/c)
        elif c==0 and s<0:
            theta = -np.pi/2
        u = rho/s
        r = theta*u
        if abs(np.linalg.norm(r)-np.pi)<1e-12 and \
        ((abs(r[0])<1e-12 and abs(r[1])<1e-12 and r[2]<0) or (abs(r[0])<1e-12 and r[1]<0) or (r[0]<0)):
            r = -r
    return r


def rodriguesResidual(x, K1, M1, p1, K2, p2):
    '''
    Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, the difference between original and estimated projections
    '''
    # Replace pass by your implementation
    P = x[:-6]
    N = int(P.shape[0]/3)
    P = P.reshape(N,3)
    P_HOMO = np.column_stack([P, np.ones(N)])
    r2 = x[-6:-3]
    t2 = x[-3:]
    C1 = K1.dot(M1)
    R2 = rodrigues(r2)
    M2 = np.column_stack([R2,t2])
    C2 = K2.dot(M2)

    p1_hat = C1.dot(P_HOMO.T)
    p1_hat = p1_hat/p1_hat[-1,:]
    p1_hat = p1_hat[0:2,:].T

    p2_hat = C2.dot(P_HOMO.T)
    p2_hat = p2_hat/p2_hat[-1,:]
    p2_hat = p2_hat[0:2,:].T

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals



def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    '''
    Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P, the optimized 3D coordinates of points
    '''    
    # Replace pass by your implementation
    t2_init = M2_init[:,-1]
    R2_init = M2_init[:,0:3]
    # print(M2_init)
    N = p1.shape[0]
    r2_init = invRodrigues(R2_init)
    x_init = np.concatenate([P_init.flatten(), r2_init, t2_init])
    # residuals = rodriguesResidual(K1, M1, p1, K2, p2, x)
    X = leastsq(rodriguesResidual, x_init, args=(K1, M1, p1, K2, p2))
    x = X[0]
    P = x[:-6]
    P = P.reshape(N,3)
    P_HOMO = np.column_stack([P, np.ones(N)])
    # print(P - P_init)

    r2 = x[-6:-3]
    t2 = x[-3:]
    R2 = rodrigues(r2)
    M2 = np.column_stack([R2, t2])

    X1 = np.zeros((N,2))
    X2 = np.zeros((N,2))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)

    for i in range(N):
        x1 = C1.dot(P_HOMO[i])
        x2 = C2.dot(P_HOMO[i])
        
        x1 = x1/x1[-1]
        x2 = x2/x2[-1]
        X1[i] = x1[0:2]
        X2[i] = x2[0:2]

    err = nlg.norm(p1-X1)**2 + nlg.norm(p2-X2)**2
    print("Optimized reprojection error is ", err)
    return M2, P 


if __name__ == "__main__":
	data = np.load('../data/some_corresp.npz')
	im1 = plt.imread('../data/im1.png')
	im2 = plt.imread('../data/im2.png')

	pts1 = data['pts1']
	pts2 = data['pts2']
	M = 640
	N = pts1.shape[0]
	random = np.random.permutation(np.arange(N))[0:7]
	# pts1 = data['pts1'][random]
	# pts2 = data['pts2'][random]
	intrinsics = np.load('../data/intrinsics.npz')
	K1 = intrinsics['K1']
	K2 = intrinsics['K2']
	F = eightpoint(pts1, pts2, M)
	# M2, P = findM2(pts1, pts2, F, K1, K2)

	# E = essentialMatrix(F,K1,K2)
	# print(E)

	# helper.displayEpipolarF(im1, im2, F[1])
	# helper.displayEpipolarF(im1,im2,F[2])
	# print(F)
	# F = eightpoint(pts1, pts2, M)
	# F, inlier = ransacF(pts1, pts2, M)

	# # # print(inlier)
	# # # np.savez("F_ransac.npz", F=F, inlier = inlier)
	# # # results = np.load("F_ransac.npz")
	# # # F = results['F']
	# # # inlier = results['inlier']
	# pts1 = pts1[inlier]
	# pts2 = pts2[inlier]
	# N = pts1.shape[0]
	# # print(pts1.shape)
	# # print(pts2.shape)
	# M2_init, P_init = findM2(pts1, pts2, F, K1, K2)
	# M1 = np.hstack([np.eye(3),np.zeros((3,1))])
	# P = P_init
	# fig1 = plt.figure()
	# ax1 = Axes3D(fig1)

	# ax1.plot(P[:,0], P[:,1], P[:,2],'bo')
	# # ax.set_zlim(3.4,4.1)
	# ax.set_ylim(-0.6,0.6)


	# # print(P_init)
	# M2, P = bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, P_init)
	# fig2 = plt.figure()
	# ax2 = Axes3D(fig2)

	# ax2.plot(P[:,0], P[:,1], P[:,2],'bo')
	# # ax.set_zlim(3.4,4.1)
	# # ax.set_ylim(-0.6,0.6)
	# plt.show()

	# F = eightpoint(pts1, pts2, M,REFINE=True)

	# F = F/F[-1,-1]
	# print(F)
	# E = essentialMatrix(F, K1, K2)
	# M2s = helper.camera2(E)

	# M2 = findM2()
	# print(M2)
	# print(M2s)
	# findM2()
	# print(M2s[:,:,idx])
	# print(F)
	# helper.displayEpipolarF(im1, im2, F)
	helper.epipolarMatchGUI(im1, im2, F)

	# # Test Rodrigues
	# r = np.array([2,2,0])
	# R = rodrigues(r)
	# print('both the below should be identity')
	# print(R.T.dot(R))
	# print(R.dot(R.T))
	# r = invRodrigues(R)
	# print('should be r')
	# print(r)

	# f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
	# ax1.imshow(im1)
	# ax2.imshow(im2)
	# ax1.set_axis_off()
	# ax2.set_axis_off()
	# # points = np.load("../data/templeCoords.npz")
	# # X1 = points['x1']
	# # Y1 = points['y1']
	# # N = X1.shape[0]
	# # X1 = X1.reshape(N)
	# # Y1 = Y1.reshape(N)
	# # pts1 = np.stack([X1, Y1], axis=1)
	# sy, sx, _ = im2.shape
	# for i in range(N):
	# 	plt.sca(ax1)
	# 	x, y = pts1[i]
	# 	# xc = int(x)
	# 	# yc = int(y)
	# 	# v = np.array([xc, yc, 1])
	# 	# l = F.dot(v)
	# 	# s = np.sqrt(l[0]**2+l[1]**2)

	# 	# if s==0:
	# 	#     error('Zero line vector in displayEpipolar')
	# 	# l = l/s
	# 	# if l[0] != 0:
	# 	#     ye = sy-1
	# 	#     ys = 0
	# 	#     xe = -(l[1] * ye + l[2])/l[0]
	# 	#     xs = -(l[1] * ys + l[2])/l[0]
	# 	# else:
	# 	#     xe = sx-1
	# 	#     xs = 0
	# 	#     ye = -(l[0] * xe + l[2])/l[1]
	# 	#     ys = -(l[0] * xs + l[2])/l[1]
	# 	# # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2)
	# 	ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
	# 	# ax2.plot([xs, xe], [ys, ye], linewidth=2)
	# 	# draw points
	# 	# x2, y2 = epipolarCorrespondence(im1, im2, F, xc, yc)
	# 	x2, y2 = pts2[i]
	# 	ax2.plot(x2, y2, 'ro', MarkerSize=3, linewidth=2)
	# 	ax2.plot(x, y, 'bo', MarkerSize=3, linewidth=2)
	# 	ax2.plot([x,x2], [y,y2],'r')
	# 	plt.draw()
	# plt.show()
	# helper.displayEpipolarF(im1, im2, F)