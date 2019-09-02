from __future__ import print_function
import torch 

x=torch.empty(5,3)
x=torch.rand(5,3)
print(x)
x=x[:,1]
print(x)