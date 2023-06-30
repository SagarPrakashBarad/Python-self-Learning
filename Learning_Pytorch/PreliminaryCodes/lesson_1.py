# how to work with tensors
import torch
import numpy as np

# x.requires_grad_(false) or x.detach()
# with torch.no_grad():

x = torch.ones(3, requires_grad=True)
print(x)

with torch.no_grad():
    y = x + 2
    print(y)
    
    
# dummy example

weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)
    
    weights.grad.zero_()
    