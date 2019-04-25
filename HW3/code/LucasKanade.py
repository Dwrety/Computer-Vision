import numpy as np
from scipy.interpolate import RectBivariateSpline 
import numpy.linalg as nlg


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input: 
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    # Parameters 
    num_iteration = 1000
    threshold = 0.01
    p = p0
    delta_p = np.array([1,1])
    x1,y1,x2,y2 = rect
    ## Inverse Compostional Algorithm
    # 1d coordinates of x and y 
    x = np.linspace(x1, x2, x2-x1+1)
    y = np.linspace(y1, y2, y2-y1+1)
    xt, yt = np.meshgrid(x, y)

    # boundary box of the coordinates
    spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    template_frame = spline.ev(yt, xt)

    # compute the derivative of the template image
    dy, dx = np.gradient(template_frame)

    # compute the warp Jacobian at W(x;0), 
    # which is constant and can be reused.
    WPJ = np.eye(2)

    # estimate the Hessian Matrix, also constant
    # numpy flatten operation vectorize the matrix
    # row-wise first, which is different from MATLAB 
    dx_vector = dx.flatten()
    dy_vector = dy.flatten()

    # Compute the steepest descent, which is delta_T*Warp_Jacobian
    sd = np.stack([dx_vector, dy_vector], axis=1)
    sd = sd.dot(WPJ)

    # Hessian Matrix
    H = sd.T.dot(sd)

    # Optimization 
    epoch = 1
    while (nlg.norm(delta_p)> threshold) and (epoch < num_iteration):
        x_coordinate = np.linspace(x1+p[0],x2+p[0],x2-x1+1)
        y_coordinate = np.linspace(y1+p[1], y2+p[1], y2-y1+1)

        X, Y = np.meshgrid(x_coordinate, y_coordinate)
        warp_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
        warp_frame = warp_spline.ev(Y, X)
        
        # compute the error, which is the sum of pixel-wise differences
        error = template_frame - warp_frame
        error = error.flatten()
        
        # Z has a shape of (N,2), it's quicker to vectorize the matrix first 
        steepest_descent = np.stack([dx_vector*error, dy_vector*error], axis=1)

        # Sum of the steepest_descent has a shape of (2, )
        sum_sd = np.sum(steepest_descent, axis=0)
        delta_p = nlg.inv(H).dot(sum_sd)
        p = p + delta_p
        epoch += 1  
        
    return p
