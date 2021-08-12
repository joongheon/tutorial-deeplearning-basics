import torch

a = torch.empty(3,1)
b = torch.empty(1,3)

def forward(x,y):
  return torch.matmul(x,y)

out_c = forward(torch.FloatTensor([[3],[2],[1]]),torch.FloatTensor([[1,2,3]]))
print(out_c.long()) 