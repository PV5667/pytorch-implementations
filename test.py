import torch
a = torch.randn(1, 64, 28, 28)
b = torch.randn(1, 32, 28, 28)

#c = a + b
c = torch.cat([a, b], dim = 1)
print(c.size())