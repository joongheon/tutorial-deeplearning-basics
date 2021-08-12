#####################################################
import torch
import numpy as np
node1 = torch.tensor(3,dtype=torch.int32)
node2 = torch.tensor(5,dtype=torch.int32)
node3 = node1+ node2
print("node1 + node2 = ",node3.numpy())
#####################################################
import torch

a = torch.empty(3,1)
b = torch.empty(1,3)

def forward(x,y):
  return torch.matmul(x,y)

out_c = forward(torch.FloatTensor([[3],[2],[1]]),torch.FloatTensor([[1,2,3]]))
print(out_c.long()) 
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
x_data = torch.FloatTensor([[1], [2], [3]])
y_data = torch.FloatTensor([[2], [4], [6]])
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.01)
epochs = 2001
for epoch in range(epochs):
    hypothesis = x_data * W + b
    cost = torch.mean((hypothesis - y_data) ** 2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, epochs, W.item(), b.item(), cost.item()))
import matplotlib.pyplot as plt
plt.plot(x_data,y_data,'ro',label='real')
plt.plot(x_data,W.item()*x_data+b.item(),label='pred')
plt.legend()
plt.show()         