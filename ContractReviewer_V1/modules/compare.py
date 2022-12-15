import torch
from modules.mlp import mlp
from torch import nn
from torch.nn import functional as F

"""
Comparing the token of a sequence with the other sequence
that's softly aligned with that token
"""
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        
        """
        Multi layer perceptron to map num_inputs to num_hiddens space
        """
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):

        """
        Comparison between A (Premise tokens) and all the Hypothesis tokens beta
        softly aligned with A. Concat operation on dimension 2 and then pass to 
        an mlp. 
        """
        V_A = self.g(torch.cat([A, beta], dim=2))

        """
        Comparison between B (Hypothesis tokens) and all the Premise tokens alpha
        softly aligned with B. Concat operation on dimension 2 and then pass to 
        an mlp. 
        """
        V_B = self.g(torch.cat([B, alpha], dim=2))

        return V_A, V_B
