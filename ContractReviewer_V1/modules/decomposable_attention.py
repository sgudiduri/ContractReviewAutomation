import torch
from torch import nn
from .attend import Attend
from .compare import Compare
from .aggregate import Aggregate
from torch.nn import functional as F

"""
Putting all the building blocks together in the following module
"""
class DecomposableAttention(nn.Module):
    def __init__(self,  embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)

        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        A, B = X['Premise'], X['Hypothesis']
        if torch.cuda.is_available():
          A = A.cuda()
          B = B.cuda()

        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        
        return Y_hat
