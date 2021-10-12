from net import OurCNN
import torch
import os
import pdb
import torch.optim as optim

model = OurCNN()

wts_model = f"{os.getcwd()}/model_wts/mnist_wts.pth" 
model.load_state_dict(torch.load(wts_model))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

grad_list = []
grad_list.append([x for x in model.parameters() if x.requires_grad != 'null'])

u,s,v = torch.pca_lowrank(grad_list[0][2])
print(torch.matmul(grad_list[0][2], v[:, :2]).shape)
pdb.set_trace()
