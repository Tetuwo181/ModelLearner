import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Softmax, Sigmoid
from typing import List


class MergedModel(torch.nn.Module):

    @staticmethod
    def build_from_paths(model_paths: List[str], class_num: int):
        models = [torch.load(model_path) for model_path in model_paths]
        return MergedModel(models, class_num)

    def __init__(self, base_models: List[torch.nn.Module], class_num: int):

        super(MergedModel, self).__init__()
        merged_output_layer_num = 0
        for index, model in enumerate(base_models):
            setattr(self, "__model" + str(index), model)
            output_layer_param_name = list(model.state_dict().keys())[-1]
            merged_output_layer_num = merged_output_layer_num + len(model.state_dict()[output_layer_param_name])
        self.__model_num = len(base_models)
        self.__class_num = class_num if class_num > 2 else 1
        self.__merged_output = Linear(merged_output_layer_num, self.__class_num)
        self.__relu = ReLU()
        self.__output_layer = Softmax(dim=1) if class_num > 2 else Sigmoid()

    def forward(self, x: Tensor) -> Tensor:

        outputs = [getattr(self, "__model"+str(index))(x) for index in range(self.__model_num)]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.__relu(outputs)
        outputs = self.__merged_output(outputs)
        output = self.__output_layer(outputs)
        return output


class WrappedMultiModelList(torch.nn.Module):

    def __init__(self, base_models: List[torch.nn.Module]):
        super(WrappedMultiModelList, self).__init__()
        self.__base_models = base_models

    def forward(self, x: Tensor):
        return [model(x) for model in self.__base_models]
