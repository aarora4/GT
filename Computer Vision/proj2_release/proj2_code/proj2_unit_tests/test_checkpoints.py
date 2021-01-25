'''
Test the presence of checkpoint files
'''
import os


def test_simple_net_checkpoint():
  assert os.path.exists(os.path.join('model_checkpoints', 'simple_net', 'checkpoint.pt')) or os.path.exists(os.path.join('..', 'model_checkpoints', 'simple_net', 'checkpoint.pt'))
