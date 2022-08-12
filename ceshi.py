import torch
from utils import anchor
a=torch.rand(2,3)
b=torch.rand(2, 2, 3)
print((b/a).shape)