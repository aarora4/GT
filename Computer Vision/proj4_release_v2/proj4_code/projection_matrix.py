import itertools
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import rq
from scipy.optimize import least_squares


def objective_func(x, **kwargs):
    """
        Calculates the difference in image (pixel coordinates) and returns 
        it as a 2*n_points vector

        Args: 
        -        x: numpy array of 11 parameters of P in vector form 
                    (remember you will have to fix P_34=1) to estimate the reprojection error
        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                    retrieve these 2D and 3D points and then use them to compute 
                    the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences betwen 
                    projected and actual 2D points

    """
    diff = None

    points_2d = kwargs['pts2d']
    points_3d = kwargs['pts3d']

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    x = np.concatenate([x, np.ones((1))], 0)
    x = np.reshape(x, (3, 4))
    pred = projection(x, points_3d)
    diff = pred - points_2d
    diff = diff.reshape((points_2d.shape[0] * 2))
    
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return diff

def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]. Your code needs to take
                       care of both cases.

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """
    projected_points_2d = None
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    points_3d = points_3d.T
    if points_3d.shape[0] == 3:
        points_3d = np.concatenate((points_3d, np.ones((1, points_3d.shape[1]))), 0)
    projected_points_2d = P @ points_3d
    projected_points_2d = projected_points_2d.T
    projected_points_2d /= projected_points_2d[:, 2][:, np.newaxis]
    projected_points_2d = projected_points_2d[:, :2]
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return projected_points_2d

def estimate_camera_matrix(pts2d: np.ndarray, 
                           pts3d: np.ndarray, 
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 
            
              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.
              
              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    start_time = time.time()
    P = None

    kwargs = {'pts2d':pts2d,
              'pts3d':pts3d}

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    initial_guess = np.reshape(initial_guess, (12))[:11]
    results = least_squares(objective_func, initial_guess, kwargs = kwargs, method = 'lm', verbose = 2, max_nfev = 50000)
    P = results.x
    P = np.concatenate([P, np.ones((1))], 0)
    P = np.reshape(P, (3, 4))
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    print("Time since optimization start", time.time() - start_time)

    return P

def decompose_camera_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix
        
        Args:
        -  P: 3x4 numpy array projection matrix
        
        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    K = None
    R = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    K, R = rq(P[:, :3])
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return K, R

def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray, 
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix
    -   K: 3x3 intrinsic matrix (numpy array)
    - R_T: 3x3 orthonormal rotation matrix (numpy array)

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    cc = None
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    result = np.linalg.inv(K @ R_T) @ P
    cc = -1 * result[:, 3]
    
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return cc

def visualize_bounding_box(P, points_3d, img):
    """
    Visualize a bounding box over the box-like item in the image.
    
    Args:
    -  P: 3x4 projection matrix
    -  points_3d : 8 x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                   or 8 x 3 array of points [X_i,Y_i,Z_i], which should be the 
                   coordinates of the bounding box's eight vertices in world 
                   coordinate system.
    -  img: A numpy array, which should be the image in which we are going to 
            visualize the bounding box.
    """
    # load and show the image
    _, ax = plt.subplots()

    ax.imshow(img)
    projected = None # your 2D projectd points
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    points = points_3d.T
    if points.shape[0] == 3:
        points = np.concatenate((points, np.ones((1, points.shape[1]))), 0)
    projected_points_2d = P @ points
    projected_points_2d = projected_points_2d.T
    projected_points_2d /= projected_points_2d[:, 2][:, np.newaxis]
    projected = projected_points_2d[:, :2]
    print(projected.shape)
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    # unit vectors in x, y, and z
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    
    # draw the bounding box
    for i, j in itertools.combinations(range(len(points_3d)), 2):
        d = points_3d[i, :] - points_3d[j, :]
        mod = np.dot(d, d)
        if any(np.square(np.dot(d, unit)) == mod for unit in [x, y, z]):
            ax.plot((projected[i, 0], projected[j, 0]), (projected[i, 1], projected[j, 1]), '-', c='green')





