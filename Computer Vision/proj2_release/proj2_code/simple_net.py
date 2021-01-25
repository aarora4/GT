import torch
import torch.nn as nn


class SimpleNet(nn.Module):
  '''Simple Network with atleast 2 conv2d layers and two linear layers.'''

  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Hints:
    1. Refer to https://pytorch.org/docs/stable/nn.html for layers
    2. Remember to use non-linearities in your network. Network without
       non-linearities is not deep.
    3. You will get 3D tensor for an image input from self.cnn_layers. You need 
       to process it and make it a compatible tensor input for self.fc_layers.
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()  # conv2d and supporting layers here
    self.fc_layers = nn.Sequential()  # linear and supporting layers here
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################
    self.cnn_layers = nn.Sequential(nn.Conv2d(1, 10, 5), nn.MaxPool2d(3, 3), nn.ReLU(), nn.Conv2d(10, 20, 5), nn.MaxPool2d(3, 3), nn.ReLU(), nn.Flatten()) 
    self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.ReLU(), nn.Linear(100, 15))
    self.loss_criterion = nn.CrossEntropyLoss()
    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ############################################################################
    # Student code begin
    ############################################################################
    model_output = self.cnn_layers(x)
    model_output = self.fc_layers(model_output)
    ############################################################################
    # Student code end
    ############################################################################

    return model_output
