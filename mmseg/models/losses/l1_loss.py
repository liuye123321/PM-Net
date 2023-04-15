
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def l1_loss(pred, target):
    # print(pred.size())
    # exit()
    if isinstance(pred, list):
        assert pred[0].size() == target[0].size() and pred[1].size() == target[1].size() \
               and pred[2].size() == target[2].size() and pred[3].size() == target[3].size()
        loss = F.l1_loss(pred[0], target[0]) + F.l1_loss(pred[1], target[1]) + F.l1_loss(pred[2], target[2]) + F.l1_loss(pred[3], target[3])
    else:
       assert pred.size() == target.size()
       loss = F.l1_loss(pred, target)

    return loss


@LOSSES.register_module
class L1_Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
