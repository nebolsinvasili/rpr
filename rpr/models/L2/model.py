import torch.nn as nn
import torch.nn.functional as F

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
        self.layer_1 = nn.Linear(in_features=3, out_features=30)
        self.layer_2 = nn.Linear(in_features=30, out_features=210)
        self.layer_3  = nn.Linear(in_features=210, out_features=1)

    def forward(self, x):
        x = F.tanh(self.layer_1(x))
        x = F.tanh(self.layer_2(x))
        x = self.layer_3(x)
        return x
    