
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights = []
    for cls_num in cls_num_list:
        per_cls_weights.append((1 - beta) / (1 - (beta ** cls_num)))         
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.1, device='cpu'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.device = device

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        
        input = input.to(self.device)
        target = target.to(self.device)
        import torch.nn.functional as F
        weight=torch.from_numpy(np.array(self.weight)).float().to(self.device)
        ce_loss = F.cross_entropy(input, target,  weight=weight)
        pt = torch.exp(-ce_loss) 
        foc_loss = ((1 -pt) ** self.gamma * ce_loss)
        loss = foc_loss.mean()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
