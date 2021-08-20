import torch
from torch import Tensor
from abc import ABC, abstractmethod


class AbstractLossCalculator(torch.nn.Module, ABC):

    def __init__(self):
        super(AbstractLossCalculator, self).__init__()

    @abstractmethod
    def forward(self, distance, y):
        pass


class AbstractLossCalculatorForInceptionV3(torch.nn.Module, ABC):

    def __init__(self):
        super(AbstractLossCalculator, self).__init__()

    @abstractmethod
    def forward(self, distance, aux_distance, y, aux_y):
        pass
