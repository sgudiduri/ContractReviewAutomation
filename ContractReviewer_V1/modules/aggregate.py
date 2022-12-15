import torch
from modules.mlp import mlp
from torch import nn
from torch.nn import functional as F

"""
Aggregate information obtained from the comparison tensors
and pass through linear layers to obtain result of classification
"""
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        
        """
        Note the use of flatten to pass to the next linear layer
        """
        self.h = mlp(num_inputs, num_hiddens, flatten=True)

        """
        Used to obtain final classification result
        """
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):

        """
        Sum both comparison sets and pass to self.h and then self.linear to 
        obtain predictions
        """
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        
        return Y_hat
