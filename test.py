import torch

s=[11,22,33,44]
for k,v in enumerate(s,2):
    print(k,v)

a = torch.randn((2,2,2))
b = torch.sum(a[:,:,0:], dim=2)
print(a)
print(b)