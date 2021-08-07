import torch
from torch import Tensor
from abc import ABC, abstractmethod


class AbstractDistanceCaluclator(torch.nn.Module, ABC):

    def __init__(self):
        super(AbstractDistanceCaluclator, self).__init__()

    @abstractmethod
    def forward(self, x0: Tensor, x1: Tensor):
        return x0 - x1
