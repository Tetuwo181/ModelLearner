from model_merger.pytorch.proc.distance.abs_calculator import AbstractDistanceCaluclator
from torch import Tensor
import torch


class L1Norm(AbstractDistanceCaluclator):

    def __init__(self):
        super(L1Norm, self).__init__()

    def forward(self, x0: Tensor, x1: Tensor):
        return self.calc_distance(x0, x1)

    def calc_distance(self, x0, x1):
        diff = super(L1Norm, self).calc_distance(x0, x1)
        try:
            return torch.sum(torch.abs(diff), 1)
        except IndexError:
            return torch.sum(torch.abs(diff))



