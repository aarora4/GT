"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
from typing import List, Tuple
from os.path import isfile, join

from os import listdir

import numpy as np
import PIL
from PIL import Image
import torch
import torch.utils.data as data
import torchvision


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args
    - path: string specifying the directory containing images
    Returns
    - images_a: list of strings specifying the paths to the images in set A,
        in lexicographically-sorted order
    - images_b: list of strings specifying the paths to the images in set B,
        in lexicographically-sorted order
    """
    images_a = []
    images_b = []

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    files = [f for f in listdir(path) if f[-3:] == 'bmp']
    files.sort()
    images_a = [join(path, files[i]) for i in range(len(files)) if i % 2 == 0 ]
    images_b = [join(path, files[i]) for i in range(len(files)) if i % 2 == 1]

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return images_a, images_b


def get_cutoff_standardddeviations(path: str) -> List[int]:
    """
    Gets the cutoff standard deviations corresponding to each pair of images.

    The cutoff are the values you discovered from experimenting in
    part 2.

    Args
    - path: string specifying the path to the .txt file with cutoff standard
      deviation values
    Returns
    - List[int]. The array should have the same
      length as the number of image pairs in the dataset
    """

    cutoffs = []

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    file1 = open(path, 'r') 
    Lines = file1.readlines()
    cutoffs = [float(line.strip()) for line in Lines]

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return cutoffs


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You can
        specify additional transforms (e.g. image resizing) if you want to, but
        it's not necessary for the images we provide you since each pair has the
        same dimensions.

        Args:
        - image_dir: string specifying the directory containing images
        - cf_file: string specifying the path to the .txt file with cutoff
          standard deviation values
        """
        images_a, images_b = make_dataset(image_dir)

        self.cutoffs = get_cutoff_standardddeviations(cf_file)

        self.transform = None

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################
        self.transform = torchvision.transforms.ToTensor()
        self.images_a = [Image.open(im) for im in images_a]
        self.images_b = [Image.open(im) for im in images_b]

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################


        return len(self.images_a)

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff standard deviation
        value at index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0 and 1.
        Make sure you transpose the dimensions so that image_a and image_b are of
        shape (c, m, n) instead of the typical (m, n, c), and convert them to
        torch Tensors.

        If you want to use a pair of images that have different dimensions from
        one another, you should resize them to match in this function using
        torchvision.transforms.

        Args
        - idx: int specifying the index at which data should be retrieved
        Returns
        - image_a: Tensor of shape (c, m, n)
        - image_b: Tensor of shape (c, m, n)
        - cutoff: int specifying the cutoff standard deviation corresponding to
          (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """

        image_a = torch.Tensor()
        image_b = torch.Tensor()
        cutoff = 0

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################

        image_a = self.transform(self.images_a[idx])
        image_b = self.transform(self.images_b[idx])
        cutoff = self.cutoffs[idx]

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        return image_a, image_b, cutoff
