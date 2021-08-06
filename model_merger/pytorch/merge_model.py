import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Softmax, Sigmoid
from typing import List


class MergedModel(torch.nn.Module):

    def __init__(self, base_models: List[torch.nn.Module], class_num: int):

        super().__init__()
        self.__base_models = base_models
        self.__output_layer = Sigmoid() if class_num < 3 else Softmax()
        self.__class_num = class_num
        self.__relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:

        outputs = [self.base_models(param) for param in x]
        outputs = torch.cat(outputs)
        outputs = self.__relu(outputs)
        outputs = Linear(len(outputs), self.__class_num)
        output = self.__output_layer(outputs)
        return output

