from model_merger.pytorch.proc.distance.abs_calculator import AbstractDistanceCaluclator
from torch import Tensor
import torch


class L1Norm(AbstractDistanceCaluclator):

    def __init__(self):
        super(L1Norm, self).__init__()

    def forward(self, x0: Tensor, x1: Tensor):
        diff = super(L1Norm, self).forward(x0, x1)
        return torch.sum(torch.abs(diff), 1)

