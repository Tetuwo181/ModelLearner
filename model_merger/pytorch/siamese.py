import torch
from torch import Tensor
from typing import Tuple, List


class SiameseNetworkPT(torch.nn.Module):

    def __init__(self, base_model: torch.nn.Module):

        super().__init__()

        self.__base_model = base_model

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:

        output1 = self.__base_model(inputs[0])
        output2 = self.__base_model(inputs[1])
        return output1, output2

    @property
    def original_model(self):
        return self.__base_model

    def get_original_predict(self, input_tensor: Tensor):
        return self.__base_model(input_tensor)