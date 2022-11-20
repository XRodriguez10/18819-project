import torch
import numpy
#import gurobipy as grb
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cnn
import xu_net

import cvxpy as cp

batch_size = 1
learning_rate = 0.01
num_epoches = 20

data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = torch.load("./xu_net_for_MNIST.pth")

for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    print('img:', img.size())
    print('label:', label)
    break
#img = img.cuda()
#print(model(img))
img = img.numpy()

weights_and_bias = []
weights = []
bias = []
'''
for name,layer in model.named_modules():
    print("name:",name)
    print("layer:",layer)
    print("type:",type(layer))
'''
for name, param in model.named_parameters():
    #print('name:',name)
    #print('param:',param)
    #print('param_np:',param.cpu().detach().numpy())
    weights_and_bias.append(param.cpu().detach().numpy())

for i in range(len(weights_and_bias)):
    if i % 2 == 0:
        weights.append(weights_and_bias[i])
    else:
        bias.append(weights_and_bias[i].reshape(1, -1))

print(len(weights))
print(len(bias))

perturb = cp.Variable((1, 784))
constraints = [-0.1 <= perturb, perturb <= 0.1]
x = img + perturb
layer_idx = 0
lb = []
ub = []
a = []
y_list = []
for name, layer in model.named_modules():
    if type(layer) is nn.Linear:
        #print("yes")
        x = x @ weights[layer_idx].T + bias[layer_idx]
        #print('x:',x)
        row, col = x.shape
        #print("row:",row)
        #print("col:",col)
        lb_temp = np.zeros_like(x, shape=x.shape)
        #print(lb_temp)
        #print(lb_temp.shape)
        ub_temp = np.zeros_like(x, shape=x.shape)
        for i in range(row):
            for j in range(col):
                obj = cp.Minimize(x[i][j])
                prob = cp.Problem(obj, constraints)
                lb_temp[i][j] = prob.solve()
                obj = cp.Maximize(x[i][j])
                prob = cp.Problem(obj, constraints)
                ub_temp[i][j] = prob.solve()
        lb.append(lb_temp)
        ub.append(ub_temp)
        print('lower bound:', lb_temp)
        print('upper bound:', ub_temp)
        #print('y:',y_list)
        #print("optimal value:",prob.solve())
        #print(x.value)
        layer_idx += 1
    elif type(layer) is nn.ReLU:
        a_temp = cp.Variable(x.shape, boolean=True)
        row, col = x.shape
        lb_temp = lb[layer_idx - 1]
        ub_temp = ub[layer_idx - 1]
        y = cp.Variable(x.shape)
        for i in range(row):
            for j in range(col):
                constraints.append(y[i][j] <= x[i][j] - lb_temp[i][j] * (1 - a_temp[i][j]))
                constraints.append(y[i][j] >= x[i][j])
                constraints.append(y[i][j] <= ub_temp[i][j] * a_temp[i][j])
                constraints.append(y[i][j] >= 0)
        x = y
        #print('reach here')
        y_list.append(y)
        a.append(a_temp)

obj = cp.Minimize(x[0][7] - x[0][2])

prob = cp.Problem(obj, constraints)
print("optimal value:", prob.solve())
#print("perturb value:",perturb.value)
img_new = img + perturb.value
img_new = torch.from_numpy(img_new).float().cuda()
print(model(torch.from_numpy(img).float().cuda()))
print(model(img_new))
