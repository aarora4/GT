'''
File: semisupervised.py
Project: Downloads
File Created: September 2020
Author: Shalini Chaudhuri (you@you.you)
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def complete_(data):
    """
    Args:
        data: N x D numpy array    
    Return:
        labeled_complete: n x D array where values contain both complete features and labels
    """
    return data[~np.isnan(data).any(axis=1)]
    
def incomplete_(data):
    """
    Args:
        data: N x D numpy array    
    Return:
        labeled_incomplete: n x D array where values contain incomplete features but complete labels
    """
    temp = data[np.isnan(data).any(axis=1)]
    return temp[~np.isnan(temp[:, -1]), :]

def unlabeled_(data):
    """
    Args:
        data: N x D numpy array    
    Return:
        unlabeled_complete: n x D array where values contain complete features but incomplete labels
    """
    temp = data[~np.isnan(data[:, :-1]).any(axis=1)]
    return temp[np.isnan(temp[:, -1]), :]


class CleanData(object):
    def __init__(self): # No need to implement
        pass
    
    def pairwise_dist(self, x, y): # [0pts] - copy from kmeans
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between 
            x[i, :] and y[j, :]
        """
        flags = np.ones(x.shape) # N X D
        flags[np.isnan(x)] = 0
        yflags = np.ones(y.shape) # M x D
        yflags[np.isnan(y)] = 0
        x2 = np.einsum('MD,ND->NM', yflags, np.nan_to_num(x)**2)
        y2 = np.einsum('ND,MD->NM', flags, np.nan_to_num(y)**2)
        xy = np.nan_to_num(x) @ np.nan_to_num(y).T
        d2 = -2 * xy + y2 + x2
        d2[d2 < 0] = 0
        return np.sqrt(d2)

    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs): # [10pts]
        """
        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points: N_complete x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_incomplete + N_complete) x (D-1) X D numpy array of length K, containing both complete points and recently filled points
            
        Hints: (1) You want to find the k-nearest neighbors within each class separately;
               (2) There are missing values in all of the features. It might be more convenient to address each feature at a time.
        """
        all_data = np.vstack((incomplete_points[:, :-1], complete_points[:, :-1])) #N x D
        interpolated = incomplete_points
        for i in range(all_data.shape[1]):
            filtered_data = all_data[~np.isnan(all_data[:, i]), :] # M x D
            dist = self.pairwise_dist(incomplete_points[:, :-1], filtered_data) #NI x M
            idx = np.argpartition(dist, K, axis = 1)[:, :K]
            rows = np.zeros(idx.shape) + np.arange(idx.shape[0])[:, np.newaxis]
            temp = np.zeros(dist.shape) #NI x M
            temp[rows.flatten().astype(int), idx.flatten()] = 1
            neighbor_mean = temp @ np.nan_to_num(filtered_data) / K
            new_col = np.zeros((incomplete_points.shape[0], 1)) # NI x 1
            neighbor_mean = np.hstack((neighbor_mean, new_col)) # NI X D + 1
            interpolated[:, i][np.isnan(interpolated[:, i])] = neighbor_mean[:, i][np.isnan(interpolated[:, i])]
        return np.vstack((interpolated, complete_points))
    


class SemiSupervised(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logits): # [0 pts] - can use same as for GMM
        """
        Args:
        logits: N x D numpy array
        """
        raise NotImplementedError

    def logsumexp(self,logits): # [0 pts] - can use same as for GMM
        """
        Args:
            logits: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logits[i,:])
        """
        raise NotImplementedError
    
    def _init_components(self, points, K, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            
        Hint: The paper describes how you should initialize your algorithm.
        """
        raise NotImplementedError

    def _ll_joint(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
            
        Hint: Assume that the three properties of the lithium-ion batteries (multivariate gaussian) are independent.  
              This allows you to treat it as a product of univariate gaussians.
        """
        raise NotImplementedError

    def _E_step(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        raise NotImplementedError

    def _M_step(self, points, gamma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
            
        Hint:  There are formulas in the slide.
        """
        raise NotImplementedError

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxD numpy array), mu and sigma.
         
        """
        raise NotImplementedError


class ComparePerformance(object):
    
    def __init__(self): #No need to implement
        pass
    
    
    def accuracy_semi_supervised(self, points, independent, n=8):
        """
        Args:
            points: Nx(D+1) numpy array, where N is the number of points in the training set, D is the dimensionality, the last column
            represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            independent: Nx(D+1) numpy array, where N is # points and D is the dimensionality and the last column are the correct labels
        Return:
            accuracy: floating number
        """
        raise NotImplementedError

    def accuracy_GNB_onlycomplete(self, points, independent, n=8):
        """
        Args:
            points: Nx(D+1) numpy array, where N is the number of only initially complete labeled points in the training set, D is the dimensionality, the last column
            represents the labels.
            independent: Nx(D+1) numpy array, where N is # points and D is the dimensionality and the last column are the correct labels
        Return:
            accuracy: floating number
        """
        raise NotImplementedError

    def accuracy_GNB_cleandata(self, points, independent, n=8):
        """
        Args:
            points: Nx(D+1) numpy array, where N is the number of clean labeled points in the training set, D is the dimensionality, the last column
            represents the labels.
            independent: Nx(D+1) numpy array, where N is # points and D is the dimensionality and the last column are the correct labels
        Return:
            accuracy: floating number
        """
        raise NotImplementedError