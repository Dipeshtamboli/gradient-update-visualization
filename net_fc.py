import torch.nn.functional as F
import torch.nn as nn
import pdb
class onlyFC(nn.Module): # Any neural generated network should be generate

  def __init__(self):
    super(onlyFC, self).__init__()

    self.fc1 = nn.Linear(28*28, 28*4)
    self.fc2 = nn.Linear(28*4, 10)

  def forward(self, x):
    # pdb.set_trace()
    x = x.view(-1, 28*28)      # x now has shape (batchsize x 432)
    x = F.relu(self.fc1(x))     # x has shape (batchsize x 10)
    x = (self.fc2(x))     # x has shape (batchsize x 10)
    return F.log_softmax(x,-1) 