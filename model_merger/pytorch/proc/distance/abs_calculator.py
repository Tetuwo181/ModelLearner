import torch
from torch import Tensor
from abc import ABC, abstractmethod


class AbstractDistanceCaluclator(torch.nn.Module, ABC):

    def __init__(self):
        super(AbstractDistanceCaluclator, self).__init__()

    @abstractmethod
    def forward(self, x0: Tensor, x1: Tensor):
        return self.calc_distance(x0, x1)

    @abstractmethod
    def calc_distance(self, x0: Tensor, x1: Tensor):
        return x0 - x1
