import glob
import os
from os import listdir
from os.path import isfile, join
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################

  files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dir_name) for f in filenames]
  files = [f for f in files if f[-3:] == 'jpg']
  images = [np.asarray(Image.open(f)) for f in files]
  
  data = [image.reshape(-1) for image in images]
  data = np.concatenate(data, 0)
  data = (data - np.min(data)) / (np.max(data) - np.min(data))
  
  mean = np.mean(data)
  std = np.std(data)
  
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
