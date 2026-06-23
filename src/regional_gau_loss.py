import warnings

import torch
from torch import nn
from torch.nn import modules as nm

from torch.nn import _reduction as _Reduction, functional as F

from .auto_weighted_loss import AutomaticWeightedLoss


class InvertImgChannels(nn.Module):
    r"""
    Produces essentially 1-input channelwise (on axies 2,3), preserving scale between the input and the output
    """

    def __init__(self):
        super().__init__()

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        # Get the maximum value per-cateogry
        imax = i.max(dim=2, keepdim=True)[0]
        imax = imax.max(dim=3, keepdim=True)[0]
        # Get the inverse by subtracting the maximum
        iinverse = imax - i

        # Scale the inverse to have the same mean value as the normal case, per-category
        iscale = i.sum(axis=(2,3), keepdim=True).div(iinverse.sum(axis=(2,3), keepdim=True))
        # Replace NaN with zeroes
        iscale[iscale != iscale] = 0
        iinverse *= iscale

        return iinverse




class RegionalGauLoss(nm.Module):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        function (str, optional): Either "mse" or "mae"

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    Examples::

        >>> loss = RegionalDenLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target, mask=0.5)
        >>> output = loss(input, target, mask=(target > 0.5))
        >>> output.backward()
    """

    VALID_REDUCTIONS = ["none", "sum", "mean"]
    VALID_FUNCTIONS  = ["mae", "mse"]

    def __init__(self, use_awl = False, positive_weighting = 10.0, reduction: str = "mean", function = "mse") -> None:
        super().__init__()

        if (not (reduction in self.VALID_REDUCTIONS)):
            raise ValueError(
                f"Invalid reduction mode: {reduction}. Expected one of {self.VALID_REDUCTIONS}."
            )

        if (not (function in self.VALID_FUNCTIONS)):
            raise ValueError(
                f"Invalid function: {function}. Expected one of {self.VALID_FUNCTIONS}."
            )

        if (use_awl and (positive_weighting != 10.0)):
            warnings.warn(f"WARNING in regional_gau_loss, use_awl TRUE, positive_weighting of {positive_weighting} will be ignored")

        if (use_awl):
            self.auto_weighted_loss = AutomaticWeightedLoss(2)
            self.weight = lambda self, lpos, lneg: self.auto_weighted_loss(lpos, lneg)
        else:
            self.positive_weighting = positive_weighting
            self.weight = lambda self, lpos, lneg: (lpos * self.positive_weighting) + lneg



        self.reduction = reduction.lower()
        self.function = function.lower()
        self.invert = InvertImgChannels()
        

    def forward(self, i: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # if self.function == "mse":
        #     loss = F.mse_loss(i, target, reduction="none")
        # else:
        #     loss = F.l1_loss(i, target, reduction="none")
        #     return loss
        
        # Mask for two terms
        # mask_pos = target > 0.0005
        # loss_pos = loss * mask_pos
        # loss_neg = loss * (mask_pos.logical_not())

        
        # Try having a positive and inverse case
        iinverse = self.invert(i)
        tinverse = self.invert(target)

        # # Reduce before combining the regions
        # if self.reduction == "sum":
        #     loss_neg = loss_neg.sum()
        #     loss_pos = loss_pos.sum()
        # elif self.reduction == "mean":
        #     loss_neg = loss_neg.mean()
        #     loss_pos = loss_pos.mean()
        # # Else no reduction

        if self.function == "mse":
            loss_pos = F.mse_loss(i, target, reduction=self.reduction)
            loss_neg = F.mse_loss(iinverse, tinverse, reduction=self.reduction)
        else:
            loss_pos = F.l1_loss(i, target, reduction=self.reduction)
            loss_neg = F.l1_loss(iinverse, tinverse, reduction=self.reduction)

        # Weight the regions
        return self.weight(self, loss_pos, loss_neg)

