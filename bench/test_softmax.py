import torch

a = torch.randn(4096, 512).half().cuda()

for i in range(10):
  torch.softmax(a, dim=-1)