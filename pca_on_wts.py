from net import OurCNN
import torch
import os
import pdb
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from net_fc import onlyFC
# model = OurCNN()
model = onlyFC()
cwd = os.getcwd()

num_epoch = 5

last_fc = torch.zeros((num_epoch, 1120))

for epoch in range(1,num_epoch+1):
    wts_model = f"{cwd}/fc_wts/cuda_net2_batch4_{epoch}.pth" 
    model.load_state_dict(torch.load(wts_model))
    grad_list = []
    grad_list.append([x for x in model.parameters() if x.requires_grad != 'null'])
    # pdb.set_trace()
    last_fc[epoch-1] = torch.reshape(grad_list[0][2], (1,-1))

u,s,v = torch.pca_lowrank(last_fc)
pca_wt = torch.matmul(last_fc, v[:, :2]).detach().numpy()
print(pca_wt.shape)

x = pca_wt[:,0]
y = pca_wt[:,1]
print(x,"\n",y)

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

plt.figure(figsize=(6,6))
plt.plot(x, y, marker="o", label=(f"x, y"))
plt.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
plt.title(f"PCA MNIST")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.legend()
plt.savefig("pca_fc.jpg")
