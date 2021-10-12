import torch.nn.functional as F
import torch.nn as nn

class OurCNN(nn.Module): # Any neural generated network should be generate

  def __init__(self):
    super(OurCNN, self).__init__()

    self.conv = nn.Conv2d(1, 3, kernel_size=5)
    self.fc1 = nn.Linear(432, 40)
    self.fc2 = nn.Linear(40, 10)

  def forward(self, x):
    x = self.conv(x)        # x now has shape (batchsize x 3 x 24 x 24)
    x = F.relu(F.max_pool2d(x,2))  # x now has shape (batchsize x 3 x 12 x 12)
    x = x.view(-1, 432)      # x now has shape (batchsize x 432)
    x = F.relu(self.fc1(x))     # x has shape (batchsize x 10)
    x = (self.fc2(x))     # x has shape (batchsize x 10)
    return F.log_softmax(x,-1) 