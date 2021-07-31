# -*- coding: utf-8 -*-
import keras.engine.training
from typing import Callable
from typing import Tuple
from typing import List
from typing import Union
from util_types import types_of_loco
from network_model.model_base import tempload
from network_model.distillation.distillation_model_builder import DistllationModelIncubator
from network_model.build_model import builder_pt, builder_with_merge
from keras.callbacks import Callback
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


def build_wrapper(img_size: types_of_loco.input_img_size = 28,
                  channels: int = 3,
                  model_name: str = "model1",
                  opt_builder: OptimizerBuilder = default_optimizer_builder,
                  loss: _Loss = CrossEntropyLoss()) -> ModelBuilder:
    """
    モデル生成をする関数を返す
    交差検証をかける際のラッパーとして使う
    :param img_size:
    :param channels:
    :param model_name:
    :param opt_builder:
    :param loss:
    :return:
    """

    def build_temp(load_path):
        base_model = tempload.builder(load_path)
        optimizer = opt_builder(base_model)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             loss)

    def build_factory(class_num):
        base_model = builder_pt(class_num, img_size, model_name)
        optimizer = opt_builder(base_model)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             loss)
    return build_temp if model_name == "tempload" else build_factory

