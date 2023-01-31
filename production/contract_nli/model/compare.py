import torch
from model.mlp import mlp
from torch import nn
from torch.nn import functional as F

class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        torch.manual_seed(123)
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        torch.manual_seed(123)
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B