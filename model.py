# importing the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# loss function
def contrastive_loss(y, t):
    nonmatch = F.relu(1 - y) # max(margin - y, 0)
    return torch.mean(t * y**2 + (1 - t) * nonmatch**2)

# Convenience function to make predictions
def predict(model, device, x1, x2):
    x1 = torch.from_numpy(x1).float().to(device)
    x2 = torch.from_numpy(x2).float().to(device)

    with torch.no_grad():
      dist = model(x1, x2).cpu().numpy()

      return dist.flatten()