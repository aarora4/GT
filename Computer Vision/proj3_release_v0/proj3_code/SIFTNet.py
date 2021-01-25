#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from torch import nn
from proj3_code.torch_layer_utils import ImageGradientsLayer


"""

Implement the SIFT Deep Net that accomplishes the identical operations as the
original SIFT algorithm (See Szeliski 4.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf


Your implementation does not need to exactly match the SIFT reference. For
example, we will be excluding scale and rotation invariance. However, here are
the key properties your descriptor should have:

(1) a 4x4 grid of cells, each feature_width/4. It is simply the
    terminology used in the feature literature to describe the spatial
    bins where gradient distributions will be described.
(2) each cell should have a histogram of the local distribution of
    gradients in 8 orientations. Appending these histograms together will
    give you 4x4 x 8 = 128 dimensions.
(3) Each feature should be normalized to unit length.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells
As described in Szeliski, a single gradient measurement creates a
weighted contribution to the 4 nearest cells and the 2 nearest
orientation bins within each cell, for 8 total contributions. This type
of interpolation probably will help, though.

Instead of explicitly computing the gradient orientation at each
pixel, we wish for you to instead filter with oriented filters (e.g. a
filter that responds to edges with a specific orientation). All of your
SIFT-like feature can be constructed entirely from filtering fairly quickly in
this way.

Regarding subgrid size -- a 4x4 filter is undesirable since it has even
dimensions, necessitating asymmetric padding along a single axis.

However, the impact is negligible -- the performance with a 5x5 filter is
identical, so we stick with the original 4x4 implementation.

You can find a review of what a conv layer does here:
    http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture05.pdf
"""

def angles_to_vectors_2d_pytorch(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to 2-d basis vectors (cosines and sines).
    You may find torch.cat(), torch.cos(), torch.sin() helpful.

    Args:
    -   angles: Torch tensor of shape (N,) representing N angles, measured in
        radians

    Returns:
    -   angle_vectors: Torch tensor of shape (N,2), representing x- and y-
        components of unit vectors in each of N angles, provided as argument.
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    x = torch.cos(angles)
    y = torch.sin(angles)
    angle_vectors = torch.stack([x, y], 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return angle_vectors


class SIFTOrientationLayer(nn.Module):
    """
    SIFT analyzes image gradients according to 8 bins, around the unit circle
    (a polar grid).
    """
    def __init__(self):
        """
        Initialize the model's layers and populate the layer weights
        appropriately. You should have 10 filters. 8 of the filters 
        correspond to the 8 orientations we want, and the last 2
        filters correspond to copying over the gradients dx, dy. 

        You may find the Pytorch function nn.Conv2d() helpful here.

        Args:
        -   None

        Returns:
        -   None
        """
        super().__init__()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        self.layer = nn.Conv2d(in_channels=2, out_channels=10, kernel_size=1, stride = 1, bias = False)
        r = (torch.arange(8)* 2 + 1) * np.pi / 8
        angles = angles_to_vectors_2d_pytorch(r)
        x = torch.zeros(2, 2)
        x[0, 0] = 1
        x[1, 1] = 1
        angles = torch.cat([angles, x], 0).view(10, 2, 1, 1)
        self.layer.weight = torch.nn.Parameter(angles)
        self.layer.bias = None
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def get_orientation_bin_weights(self) -> torch.nn.Parameter():
        """
        Populate the conv layer weights according to orientation basis.

        A 1x1 convolution layer makes perfect sense. For example, consider a
        1x1 CONV with 32 filters. Suppose your input is (1,64,56,56) in NCHW
        order. Then each filter has size (64,1,1) and performs a 64-dimensional
        dot product, producing a (1,32,56,56) tensor. In other words, you are
        performing a dot-product of two vectors with dim-64, and you do this
        with 32 different bases. This can be thought of as a 32x64 weight
        matrix.

        The orientations you choose should align with the following angles
        in this order:
        pi/8, 3pi/8, .... 15pi/8

        Args:
        -   None

        Returns:
        -   weight_param: Torch nn.Parameter, containing (10,2) matrix for the
            1x1 convolution's dot product
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        self.layer = nn.Conv2d(in_channels=2, out_channels=10, kernel_size=1, stride = 1, bias = False)
        r = (torch.arange(8)* 2 + 1) * np.pi / 8
        angles = angles_to_vectors_2d_pytorch(r)
        x = torch.zeros(2, 2)
        x[0, 0] = 1
        x[1, 1] = 1
        angles = torch.cat([angles, x], 0).view(10, 2, 1, 1)
        self.layer.weight = torch.nn.Parameter(angles)
        self.layer.bias = None
        weight_param = self.layer.weight
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return weight_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the SIFTOrientationLayer().

        Args:
        -   x: Torch tensor with shape (1,2,m,n)

        Returns:
        -   out: Torch tensor with shape (1,10,m,n)
        """
        return self.layer(x)



class HistogramLayer(nn.Module):
    def __init__(self) -> None:
        """
        Initialize parameter-less histogram layer, that accomplishes
        per-channel binning.

        Args:
        -   None

        Returns:
        -   None
        """
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        """
        Complete a feedforward pass of the histogram/binning layer byforming a
        weighted histogram at every pixel value.

        The input should have 10 channels, where the first 8 represent cosines
        values of angles between unit circle basis vectors and image gradient
        vectors, at every pixel. The last two channels will represent the
        (dx, dy) coordinates of the image gradient at this pixel.

        The weighted histogram can be created by elementwise multiplication of
        a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
        tensor, where a tensor cell is activated if its value represents the
        maximum channel value within a "fibre" (see
        http://cs231n.github.io/convolutional-networks/ for an explanation of a
        "fibre"). There will be a fibre (consisting of all channels) at each of
        the (M,N) pixels of the "feature map".

        The four dimensions represent (N,C,H,W) for batch dim, channel dim,
        height dim, and weight dim, respectively. Our batch size will be 1.

        In order to create the 4d binary occupancy tensor, you may wish to
        index in at many values simultaneously in the 4d tensor, and read or
        write to each of them simultaneously. This can be done by passing a 1d
        Pytorch Tensor for every dimension, e.g. by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

        You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
        flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
        torch.norm() helpful.

        With a double for-loop you could expect 20 sec. runtime for this
        function. You may not submit code with a triple for-loop (which would
        take over 60 seconds). With tensor indexing, this should take 0.08-0.11
        sec.

        Args:
        -   x: tensor with shape (1,10,M,N), where M,N are height, width

        Returns:
        -   per_px_histogram: tensor with shape (1,8,M,N) representing a weighted
            histogram at every pixel
        """
        cosines = x[:,:8,:,:] # Contains gradient contributions in each direction
        im_grads = x[:,8:,:,:] # Contains dx, dy

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        weights = torch.sqrt(im_grads[:, 0, :, :] * im_grads[:, 0, :, :] + im_grads[:, 1, :, :] * im_grads[:, 1, :, :])
        max, indices = torch.max(cosines, 1, keepdim=True)
        max = max.repeat(1, 8, 1, 1)
        #torch.cat([max for _ in range(8)], 1)
        mask = torch.zeros_like(cosines)
        mask[max == cosines] = 1
        per_px_histogram = mask * weights

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return per_px_histogram


class SubGridAccumulationLayer(nn.Module):
    """
    """
    def __init__(self) -> None:
        """
        Given 8-dimensional feature vectors at each pixel, accumulate features
        over 4x4 subgrids.

        You may find the Pytorch function nn.Conv2d() helpful here. In Pytorch,
        a Conv2d layer's behavior is governed by the `groups` parameter. You
        will definitely need to understand the effect of this parameter. With
        groups=1, if your input is 28x28x8, and you wish to apply a 5x5 filter,
        then you will be convolving all inputs to all outputs (i.e. you will be
        convolving a 5x5x8 filter at every possible location over the feature
        map. However, if groups=8, then you will be convolving a 5x5x1 filter
        over each channel separately. You will also need to pad by 2 on every
        side to allow for every pixel to have a window around it.

        Args:
        -   None

        Returns:
        -   None
        """
        super().__init__()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        self.layer = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4, 4), stride = (1, 1), bias = False, groups = 8, padding = 2)
        self.layer.weight = nn.Parameter(torch.ones(8,1, 4, 4))
        self.layer.bias = None
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the SubGridAccumulationLayer().

        Args:
        -   x: Torch tensor representing an (b,c,m,n) layer, where b=1, c=8

        Returns:
        -   out: Torch tensor representing an (b,c,m,n) layer, where b=1, c=8
        """
        output = self.layer(x)
        return self.layer(x)




class SIFTNet(nn.Module):

    def __init__(self):
        """
        See http://cs231n.github.io/convolutional-networks/ for more details on
        what a conv layer does.

        We Create a nn.Sequential() network, using the 4 specific layers you have
        implemented above. The layers above are not in any particular order.

        Args:
        -   None

        Returns:
        -   None
        """
        super().__init__()


        # (1) Conv layer to compute image gradients with two directed Sobel filters.
        # (2) Conv layer to project image gradients along 8 basis directions, and to
        #   replicate the gradient values for the subsequent histogram computation.
        # (3) Custom HistogramLayer
        # (4) Conv layer to accumulate 4x4 neighborhoods.

        self.net = nn.Sequential(
            ImageGradientsLayer(),
            SIFTOrientationLayer(),
            HistogramLayer(),
            SubGridAccumulationLayer()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SIFTNet. Feed a grayscale image through the SIFT
        network to obtain accumulated gradient histograms at every single
        pixel.

        Args:
        -   x: Torch tensor of shape (1,1,M,N) representing single grayscale
            image.

        Returns:
        -   Torch tensor representing 8-bin weighted histograms, accumulated
            over 4x4 grid cells
        """
        return self.net(x)


def get_sift_subgrid_coords(x_center: int, y_center: int):
    """
    Given the center point of a 16x16 patch, we eventually want to pull out the
    accumulated values for each of the 16 subgrids. We need the coordinates to
    do so, so return the 16 x- and y-coordinates, one for each 4x4 subgrid.

    Args:
    -   x_center: integer representing x-coordinate of keypoint.
    -   y_center: integer representing y-coordinate of keypoint.

    Returns:
    -   x_grid: (16,) representing x-coordinates
    -   y_grid: (16,) representing y-coordinates.
    """

    x = np.linspace(x_center-6, x_center+6, 4)
    y = np.linspace(y_center-6, y_center+6, 4)
    x_grid, y_grid = np.meshgrid(x, y)

    x_grid = x_grid.flatten().astype(np.int64)
    y_grid = y_grid.flatten().astype(np.int64)

    return x_grid, y_grid


def get_siftnet_features(img_bw: torch.Tensor, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Given a list of (x,y) coordinates, pull out the SIFT features within the
    16x16 neighborhood around each (x,y) coordinate pair.

    Then normalize each 128-dimensional vector to have unit length.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one. Please raise each
    feature vector to the 0.9 power after normalizing.

    Args:
    -   img_bw: Torch tensor with shape (1,1,M,N) representing grayscale image.
    -   x: Numpy array with shape (K,)representing x-coordinates
    -   y: Numpy array with shape (K,)representing y-coordinates

    Returns:
    -   fvs: feature vectors of shape (K,128)
    """
    assert img_bw.shape[0] == 1
    assert img_bw.shape[1] == 1
    assert img_bw.dtype == torch.float32

    net = SIFTNet()
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    m = nn.ZeroPad2d(12)
    sift = m(net(img_bw))
    
    a = np.linspace(-6, 6, 4)
    b = np.linspace(-6, 6, 4)
    x_grid, y_grid = np.meshgrid(a, b)
    x_grid = np.expand_dims(x_grid, 0)
    x_grid = np.tile(x_grid, (x.shape[0], 1, 1))
    y_grid = np.expand_dims(y_grid, 0)
    y_grid = np.tile(y_grid, (y.shape[0], 1, 1))
    x_grid += np.expand_dims(x, (1, 2))
    y_grid += np.expand_dims(y, (1, 2))
    
    x_grid = x_grid.flatten().astype(np.int64)
    y_grid = y_grid.flatten().astype(np.int64)
    
    s = sift[0, :, y_grid + 12, x_grid + 12]
    s = s.transpose(0, 1)
    s = s.view(x.shape[0], -1, 8)
    fvs = s.reshape(-1, 128)
    #fvs = fvs / torch.sqrt(torch.sum(fvs * fvs, 1)).view(-1, 1)
    fvs = fvs / torch.sum(fvs, 1).view(-1, 1)
    fvs[fvs < 0] = 0
    fvs = fvs ** 0.9
    fvs = fvs.detach().numpy()
    fvs = fvs[~np.isnan(fvs).any(axis=1)]
    
    '''
    features = []
    
    for i in range(x.shape[0]):
        xidx, yidx = get_sift_subgrid_coords(x[i], y[i])
        s = sift[0, :, yidx + 12, xidx + 12].squeeze()
        s = torch.transpose(s, 0, 1)
        s = s.reshape(128)
        features.append(s)
    fvs = torch.stack(features, 0)
    fvs = fvs / torch.sqrt(torch.sum(fvs * fvs, 1)).view(-1, 1)
    fvs[fvs < 0] = 0
    fvs = fvs ** 0.9
    fvs = fvs.detach().numpy()
    fvs = fvs[~np.isnan(fvs).any(axis=1)]
    '''
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
