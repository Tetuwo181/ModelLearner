# -*- coding: utf-8 -*-
from __future__ import annotations
from tensorflow.keras import Model
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
from model_merger.pytorch.proc.distance.calculator import L1Norm
from model_merger.pytorch.proc.distance.abs_calculator import AbstractDistanceCaluclator
from model_merger.pytorch.proc.loss.calculator import AAEUMLoss
from model_merger.pytorch.proc.loss.abstract_calculator import AbstractLossCalculator
from model_merger.pytorch.proc.shiamese_loss import SiameseLoss, SiameseLossForInceptionV3
from model_merger.pytorch.siamese import SiameseNetworkPT


ModelBuilderResult = Model | list[Callback]

ModelBuilder = Callable[[int], ModelBuilderResult] | Callable[[str | tuple[str, str]], Model] | DistllationModelIncubator

OptimizerBuilder = Callable[[Module], Optimizer]


def optimizer_builder(optimizer, **kwargs):
    def build(base_model: Module):
        kwargs["params"] = base_model.parameters()
        return optimizer(**kwargs)
    return build


default_optimizer_builder = optimizer_builder(SGD)


class PytorchModelBuilder(object):

    def __init__(self,
                 img_size: types_of_loco.input_img_size = 28,
                 channels: int = 3,
                 model_name: str = "model1",
                 opt_builder: OptimizerBuilder = default_optimizer_builder,
                 loss: _Loss = None,
                 decide_dataset_generator=None,
                 nearest_data_ave_num=1,
                 will_calc_rate_real_data_train=False):
        self.__img_size = img_size
        self.__channels = channels
        self.__model_name = model_name
        self.__opt_builder = opt_builder
        self.__loss = loss
        self.__decide_dataset_generator = decide_dataset_generator
        self.__nearest_data_ave_num = nearest_data_ave_num
        self.__will_calc_rate_real_data_train = will_calc_rate_real_data_train

    def build_raw_model(self, model_builder_input) -> torch.nn.Module:
        if self.__model_name == "tempload":
            return torch.jit.load(model_builder_input)
        return builder_pt(model_builder_input, self.__img_size, self.__model_name)

    def build_model_builder_wrapper(self, model_builder_input):
        base_model = self.build_raw_model(model_builder_input)
        optimizer = self.__opt_builder(base_model)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             self.__loss,
                                             decide_dataset_generator=self.__decide_dataset_generator,
                                             nearest_data_ave_num=self.__nearest_data_ave_num,
                                             will_calc_rate_real_data_train=self.__will_calc_rate_real_data_train)

    def __call__(self, model_builder_input):
        return self.build_model_builder_wrapper(model_builder_input)


class PytorchSiameseModelBuilder(PytorchModelBuilder):

    def __init__(self,
                 q: float,
                 img_size: types_of_loco.input_img_size = 28,
                 channels: int = 3,
                 model_name: str = "model1",
                 opt_builder: OptimizerBuilder = default_optimizer_builder,
                 loss_calculator: AbstractLossCalculator = None,
                 calc_distance: AbstractDistanceCaluclator=L1Norm(),
                 is_inceptionv3: bool = False,
                 decide_dataset_generator=None,
                 nearest_data_ave_num=1,
                 will_calc_rate_real_data_train=False):
        use_loss_calculator = AAEUMLoss(q) if loss_calculator is None else loss_calculator
        loss = SiameseLossForInceptionV3(calc_distance, use_loss_calculator) if is_inceptionv3 else SiameseLoss(calc_distance, use_loss_calculator)
        super(PytorchSiameseModelBuilder, self).__init__(img_size,
                                                         channels,
                                                         model_name,
                                                         opt_builder,
                                                         loss,
                                                         decide_dataset_generator,
                                                         nearest_data_ave_num,
                                                         will_calc_rate_real_data_train
                                                         )

    def build_raw_model(self, model_builder_input) -> torch.nn.Module:
        original_model = super(PytorchSiameseModelBuilder, self).build_raw_model(model_builder_input)
        return SiameseNetworkPT(original_model)

