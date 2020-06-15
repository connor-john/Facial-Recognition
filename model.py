# importing the libraries
import torch
import torch.nn as nn

# Siamese Neural Network
class SiameseModel(nn.Module):
  def __init__(self, feature_dim):
    super(SiameseModel, self).__init__()

    # CNN
    self.cnn = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1*1*64, 128), # TODO: calculate convolutional arith.
        nn.ReLU(),
        nn.Linear(128, feature_dim),
    )

  def forward(self, i1, i2):
    feat1 = self.cnn(i1)
    feat2 = self.cnn(i2)

    # return euclid. distance between the features
    return torch.norm(feat1 - feat2, dim = -1)