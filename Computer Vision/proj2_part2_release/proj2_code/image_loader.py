'''
Script with Pytorch's dataloader class
'''

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image


class ImageLoader(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self,
               root_dir: str,
               split: str = 'train',
               transform: torchvision.transforms.Compose = None):
    '''
    Init function for the class.

    Note: please load data only for the mentioned split.

    Args:
    - root_dir: the dir path which contains the train and test folder
    - split: 'test' or 'train' split
    - transforms: the transforms to be applied to the data
    '''
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    if split == 'train':
      self.curr_folder = os.path.join(root_dir, self.train_folder)
    elif split == 'test':
      self.curr_folder = os.path.join(root_dir, self.test_folder)

    self.class_dict = self.get_classes()
    self.dataset = self.load_imagepaths_with_labels(self.class_dict)

  def load_imagepaths_with_labels(self,
                                  class_labels: Dict[str, int]
                                  ) -> List[Tuple[str, int]]:
    '''
    Fetches all image paths along with labels

    Args:
    -   class_labels: the class labels dictionary, with keys being the classes
                      in this dataset and the values being the class index.
    Returns:
    -   list[(filepath, int)]: a list of filepaths and their class indices
    '''

    img_paths = []  # a list of (filename, class index)

    ############################################################################
    # Student code begin
    ############################################################################
    for key in self.class_dict.keys():
        path = os.path.join(self.curr_folder, key)
        img_paths.extend([(os.path.join(path, f), self.class_dict[key]) for f in os.listdir(path) if f[-3:] == 'jpg'])
    
    ############################################################################
    # Student code end
    ############################################################################

    return img_paths

  def get_classes(self) -> Dict[str, int]:
    '''
    Get the classes (which are folder names in self.curr_folder) along with
    their associated integer index.

    Note: Assign integer indicies 0-14 to the 15 classes.

    Returns:
    -   Dict of class names (string) to integer labels
    '''

    classes = dict()
    ############################################################################
    # Student code begin
    ############################################################################
    classes = {'bedroom': 0, 'coast': 1, 'forest':2, 'highway':3, 'industrial':4, 
        'insidecity':5, 'kitchen':6, 'livingroom':7, 'mountain':8, 'office':9, 
        'opencountry':10, 'store':11, 'street':12, 'suburb':13, 'tallbuilding':14}
    ############################################################################
    # Student code end
    ############################################################################

    return classes

  def load_img_from_path(self, path: str) -> Image:
    ''' 
    Loads the image as grayscale (using Pillow)

    Note: do not normalize the image to [0,1]

    Args:
    -   path: the path of the image
    Returns:
    -   image: grayscale image loaded using pillow (Use 'L' flag while converting using Pillow's function)
    '''

    img = None
    ############################################################################
    # Student code begin
    ############################################################################

    img = Image.open(path)
    ############################################################################
    # Student code end
    ############################################################################

    return img

  def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
    '''
    Fetches the item (image, label) at a given index

    Note: Do not forget to apply the transforms, if they exist

    Hint:
    1) get info from self.dataset
    2) use load_img_from_path
    3) apply transforms if valid

    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    '''
    img = None
    class_idx = None

    ############################################################################
    # Student code start
    ############################################################################
    path, class_idx = self.dataset[index]
    img = self.load_img_from_path(path)
    img = self.transform(img)
    

    ############################################################################
    # Student code end
    ############################################################################

    return img, class_idx

  def __len__(self) -> int:
    """
    Returns the number of items in the dataset

    Returns:
        int: length of the dataset
    """

    l = 0

    ############################################################################
    # Student code start
    ############################################################################
    l = len(self.dataset)

    ############################################################################
    # Student code end
    ############################################################################

    return l
