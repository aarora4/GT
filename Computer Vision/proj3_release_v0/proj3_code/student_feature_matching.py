from typing import Tuple

import numpy as np


def compute_feature_distances(features1: np.ndarray, 
                              features2: np.ndarray) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    x2 = np.sum(features1**2, 1)
    y2 = np.sum(features2**2, 1)
    xy = features1 @ features2.T
    d2 = -2 * xy + y2 + x2[:, np.newaxis]
    d2[d2 < 0] = 0
    dists = np.sqrt(d2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1: np.ndarray, 
                   features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform nearest-neighbor matching with ratio test.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    The results should be sorted in descending order of confidence.

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)


    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    dist = compute_feature_distances(features1, features2)
    m = np.min(dist, 1)
    i = np.argmin(dist, 1)
    matches = np.stack([np.arange(features1.shape[0]), i], 1)
    idx = np.argsort(m)
    sorted = np.sort(dist, 1)
    confidences =  sorted[:, 0]
    matches = matches[idx, :]
    confidences = confidences[idx]
    partial = sorted[idx, :]
    matches = matches[partial[:, 1] > 0, :]
    confidences = confidences[partial[:, 1] > 0]
    s = partial[partial[:, 1] > 0, 0]/ partial[partial[:, 1] > 0, 1]
    t = 0.7
    matches = matches[s < t]
    confidences = confidences[s < t]
    
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
