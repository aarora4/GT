import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from numpy import linalg as LA


class Isomap(object):
    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [3 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
                """

        x2 = np.sum(x**2, 1)
        y2 = np.sum(y**2, 1)
        xy = x @ y.T
        d2 = -2 * xy + y2 + x2[:, np.newaxis]
        d2[d2 < 0] = 0
        return np.sqrt(d2)

    def manifold_distance_matrix(self, x, K):  # [10 pts]
        """
        Args:
            x: N x D numpy array
        Return:
            dist_matrix: N x N numpy array, where dist_matrix[i, j] is the euclidean distance between points if j is in the neighborhood N(i)
            or comp_adj = shortest path distance if j is not in the neighborhood N(i).
        Hint: After creating your k-nearest weighted neighbors adjacency matrix, you can convert it to a sparse graph
        object csr_matrix (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) and utilize
        the pre-built Floyd-Warshall algorithm (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.floyd_warshall.html)
        to compute the manifold distance matrix.
        """
        N, D = x.shape
        d2 = self.pairwise_dist(x, x)
        a = np.tile(np.arange(N).reshape(N, 1), (1, N - K - 1)).flatten()
        b = np.argsort(d2,axis =  1)[:,K+1:].flatten()
        d2[a, b] = 0
        sparse = csr_matrix(d2)
        dist_matrix = floyd_warshall(csgraph=sparse, directed=False, return_predecessors=False)
        return dist_matrix

    def multidimensional_scaling(self, dist_matrix, d):  # [10 pts]
        """
        Args:
            dist_matrix: N x N numpy array, the manifold distance matrix
            d: integer, size of the new reduced feature space
        Return:
            S: N x d numpy array, X embedding into new feature space.
        """
        N, _ = dist_matrix.shape
        C = np.identity(N) - (1 / N)
        B = -0.5 * C @ (dist_matrix ** 2) @ C
        w, v = np.linalg.eigh(B)
        print(w.shape, v.shape)
        idx = np.argsort(w)[::-1] [:d]
        w = w[idx]
        print(w)
        v = v[:, idx]
        return v * np.sqrt(w)

    # you do not need to change this
    def __call__(self, data, K, d):
        dist_matrix = self.manifold_distance_matrix(data, K)
        return self.multidimensional_scaling(dist_matrix, d)