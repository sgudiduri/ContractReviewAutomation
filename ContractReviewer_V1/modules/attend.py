import torch
from modules.mlp import mlp
from torch import nn
from torch.nn import functional as F

"""
Used to align tokens in one text sequence to another
"""
class Attend(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        
        """
        Multi layer perceptron to map num_inputs to num_hiddens space
        """
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):

        """
        Shape of A: (Batch Size, Tokens in A, Emb Size)
        Shape of B: (Batch Size, Tokens in B, Emb Size)

        Linear Layer Mapping
        Shape of f_A: (Batch Size, Tokens in A, Hidden Size)
        Shape of f_B: (Batch Size, Tokens in B, Hidden Size)
        """
        f_A = self.f(A)
        f_B = self.f(B)

        """
        Shape of e: (Batch Size, Tokens in A, Hidden Size) @ (Batch Size, Tokens in B, Hidden Size).T 
                =   (Batch Size, Tokens in A, Hidden Size) @ (Batch Size, Hidden Size, Tokens in B) 
                =   (Batch Size, Tokens in A, Tokens in B)
        """
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))

        """
        Shape of beta: (Batch Size, Tokens in A, Emb Size)
        B (Hypothesis) is softly aligned with every token in A (Premise) 
        """
        beta = torch.bmm(F.softmax(e, dim=-1), B)

        """
        Shape of alpha: (Batch Size, Tokens in B, Emb Size)
        A (Premise) is softly aligned with every token in B (Hypothesis)
        """
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)

        return beta, alpha
