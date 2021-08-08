# -*- coding: utf-8 -*-
import keras.engine.training
from typing import Callable
from typing import Tuple
from typing import List
from typing import Union
from util_types import types_of_loco
from network_model.distillation.distillation_model_builder import DistllationModelIncubator
from network_model.build_model import builder_pt, builder_with_merge
from keras.callbacks import Callback
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss, Module
from network_model.wrapper.pytorch.model_pt import ModelForPytorch


ModelBuilderResult = Union[keras.engine.training.Model, List[Callback]]

ModelBuilder = Union[Callable[[int], ModelBuilderResult],
                     Callable[[Union[str, Tuple[str, str]]], keras.engine.training.Model],
                     DistllationModelIncubator]

OptimizerBuilder = Callable[[Module], Optimizer]


def optimizer_builder(optimizer, **kwargs):
    def build(base_model: Module):
        kwargs["params"] = base_model.parameters()
        return optimizer(**kwargs)
    return build


default_optimizer_builder = optimizer_builder(SGD)


class PytorchModelBuilder:

    def __init__(self,
                 img_size: types_of_loco.input_img_size = 28,
                 channels: int = 3,
                 model_name: str = "model1",
                 opt_builder: OptimizerBuilder = default_optimizer_builder,
                 loss: _Loss = None):
        self.__img_size = img_size
        self.__channels = channels
        self.__model_name = model_name
        self.__opt_builder = opt_builder
        self.__loss = loss

    def build_temp(self, load_path):
        base_model = torch.jit.load(load_path)
        optimizer = self.__opt_builder(base_model)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             self.__loss)

    def build_factory(self, class_num):
        base_model = builder_pt(class_num, self.__img_size, self.__model_name)
        optimizer = self.__opt_builder(base_model)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             self.__loss)

    def __call__(self, model_builder_input):
        if self.__model_name == "tempload":
            return self.build_temp(model_builder_input)
        return self.build_factory(model_builder_input)

