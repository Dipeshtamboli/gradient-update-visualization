import pdb
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch
import torchvision

from net import OurCNN
from net_fc import onlyFC
from train_test import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# pdb.set_trace()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.1307,),(0.3081,))])

cwd = os.getcwd()
# pdb.set_trace()
train_dataset = torchvision.datasets.MNIST(cwd+'/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(cwd+'/data', train=False, download=True, transform=transform)

batch_size_train, batch_size_test = 4, 1000

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

batch_idx, (images, targets) = next(enumerate(train_loader))
# print(f'current batch index is {batch_idx}')
# print(f'images has shape {images.size()}')
# print(f'targets has shape {targets.size()}')


# fig, ax = plt.subplots(3,3)
# fig.set_size_inches(12,12)
# for i in range(3):
#   for j in range(3):
#     ax[i,j].imshow(images[i*3+j][0], cmap='gray')
#     ax[i,j].set_title(f'label {targets[i*3+j]}')
# fig.savefig("mnist.jpg")




# classifier = OurCNN()
classifier = onlyFC()
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.8)

import torch.optim as optim

max_epoch = 5

classifier = classifier.to(device)
for epoch in range(1, max_epoch+1):
    train(classifier, epoch, train_loader, optimizer, verbose = False)
    test(classifier, epoch, test_loader, optimizer)

    PATH = f"{cwd}/fc_wts/{device}_net2_batch{batch_size_train}_{epoch}.pth" 
    torch.save(classifier.state_dict(), PATH)