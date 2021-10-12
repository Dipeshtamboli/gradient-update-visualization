from net import OurCNN
import torch
import os
import pdb
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
model = OurCNN()
cwd = os.getcwd()
num_epoch = 5
last_fc = torch.zeros((num_epoch, 400))

for epoch in range(1,num_epoch+1):
    wts_model = f"{cwd}/model_wts/cuda_net2_batch4_{epoch}.pth" 
    model.load_state_dict(torch.load(wts_model))
    grad_list = []
    grad_list.append([x for x in model.parameters() if x.requires_grad != 'null'])
    
    last_fc[epoch-1] = torch.reshape(grad_list[0][4], (1,-1))

u,s,v = torch.pca_lowrank(last_fc)
pca_wt = torch.matmul(last_fc, v[:, :2]).detach().numpy()
print(pca_wt.shape)
# pdb.set_trace()



plt.figure(figsize=(6,6))
plt.scatter(pca_wt[:,0],pca_wt[:,1], s=5, label=(f"x, y"))
plt.title(f"PCA MNIST")
# plt.xlim(-5,5)
# plt.ylim(-5,5)
# plt.xticks(range(-5,6))
# plt.yticks(range(-5,6))
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.legend()
plt.savefig("pca.jpg")
