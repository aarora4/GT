from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N (*3 for color images)
            S: min(N, D) * 1 (* 3 for color images)
            V: D * D (* 3 for color images)
        """
        color = (len(X.shape) == 3)
        if not color:
            X = np.expand_dims(X, 2)
        U = []
        S = []
        V = []
        for i in range(X.shape[2]):
            u, s, v = np.linalg.svd(X[:, :, i])
            U.append(u)
            S.append(s)
            V.append(v)
        U = np.stack(U, 2)
        S = np.stack(S, 1)
        V = np.stack(V, 2)
        if not color:
            U = U[:, :, 0]
            V = V[:, :, 0]
            S = S[:, 0]
        S = S[:min(X.shape[0], X.shape[1])]
        return (U, S, V)
        


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        color = (len(U.shape) == 3)
        if not color:
            U = np.expand_dims(U, 2)
            S = np.expand_dims(S, 1)
            V = np.expand_dims(V, 2)
        X = []
        for i in range(U.shape[2]):
            u = U[:, :k, i]
            s = S[:k, i]
            v = V[:k, :, i]
            s_mat = np.zeros((k, k))
            for j in range(k):
                s_mat[j, j] = s[j]
            x = u @ s_mat @ v
            X.append(x)
        X = np.stack(X, 2)
        if not color:
            X = X[:, :, 0]
        return X

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        return (k * X.shape[0] + k * X.shape[1] + k) / (X.shape[1] * X.shape[0])

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        color = (len(S.shape) == 2)
        if not color:
            S = np.expand_dims(S, 1)
        var = []
        for i in range(S.shape[1]):
            var.append(np.sum(S[:k, i] ** 2) / np.sum(S[:, i] ** 2))
        if not color:
            return var[0]
        return var