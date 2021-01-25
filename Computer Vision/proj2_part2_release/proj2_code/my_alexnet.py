import torch
import torch.nn as nn
from torchvision.models import alexnet
'''
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
'''            
class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
    weights and biases of a layer to not require gradients.

    Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

    Note: Remove the last linear layer in Alexnet and add your own layer to 
    perform 15 class classification.

    Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################
    #with suppress_stdout_stderr():
    alex = alexnet(pretrained=True)
    self.cnn_layers = alex.features
    self.fc_layers = alex.classifier
    self.fc_layers[-1] = nn.Linear(4096, 15)
    self.loss_criterion = nn.CrossEntropyLoss()
    
    for param in self.cnn_layers.parameters():
        param.requires_grad = False
    for param in self.fc_layers.parameters():
        param.requires_grad = False
    for param in self.fc_layers[-1].parameters():
        param.requires_grad = True
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
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    ############################################################################
    model_output = self.cnn_layers(x).view(-1, 9216)
    model_output = self.fc_layers(model_output)
    ############################################################################
    # Student code end
    ############################################################################

    return model_output
