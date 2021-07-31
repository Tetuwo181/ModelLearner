# -*- coding: utf-8 -*-
import keras.engine.training
from typing import Callable
from typing import Tuple
from typing import List
from typing import Union
from util_types import types_of_loco
from network_model.model_base import tempload
from network_model.distillation.distillation_model_builder import DistllationModelIncubator
from network_model.build_model import builder, builder_with_merge
from keras.callbacks import Callback
from torch.optim.optimizer import Optimizer
from torch.optim import SGD
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
from network_model.wrapper.pytorch.model_pt import ModelForPytorch

ModelBuilderResult = Union[keras.engine.training.Model, List[Callback]]

ModelBuilder = Union[Callable[[int], ModelBuilderResult],
                     Callable[[Union[str, Tuple[str, str]]], keras.engine.training.Model],
                     DistllationModelIncubator]


def init_input_image(size: types_of_loco.input_img_size):
    def builder_of_generator(class_num: int, channels: int = 1, optimizer: Optimizer = SGD()):
        """
        Ganのgenerator部を作成する
        :param class_num
        :param channels:色の出力変数（白黒画像なら1）
        :param optimizer: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
        :return: discriminator部のモデル
        """
        return builder(class_num, size, channels, optimizer)
    return builder_of_generator


def build_wrapper(img_size: types_of_loco.input_img_size = 28,
                  channels: int = 3,
                  model_name: str = "model1",
                  optimizer: Optimizer = SGD(),
                  loss: _Loss = CrossEntropyLoss()) -> ModelBuilder:
    """
    モデル生成をする関数を返す
    交差検証をかける際のラッパーとして使う
    :param img_size:
    :param channels:
    :param model_name:
    :param optimizer:
    :param loss:
    :return:
    """

    def build_temp(load_path):
        base_model = tempload.builder(load_path)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             loss)

    def build_factory(class_num):
        base_model = builder(class_num, img_size, channels, optimizer, model_name)
        return ModelForPytorch.build_wrapper(base_model,
                                             optimizer,
                                             loss)
    return build_temp if model_name == "tempload" else build_factory

