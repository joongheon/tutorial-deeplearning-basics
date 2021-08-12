import torch
import numpy as np
node1 = torch.tensor(3,dtype=torch.int32)
node2 = torch.tensor(5,dtype=torch.int32)
node3 = node1+ node2
print("node1 + node2 = ",node3.numpy())

