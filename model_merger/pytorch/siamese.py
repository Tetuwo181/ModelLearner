import torch
from torch import Tensor
from typing import Tuple


class SiameseNetworkPT(torch.nn.Module):

    def __init__(self, base_model: torch.nn.Module):

        super().__init__()

        self.__base_model = base_model

    def forward(self, input1: Tensor, input2: Tensor) -> Tuple[Tensor, Tensor]:

        output1 = self.__base_model(input1)
        output2 = self.__base_model(input2)
        return output1, output2
