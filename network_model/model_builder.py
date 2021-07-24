# -*- coding: utf-8 -*-
import keras.engine.training
from typing import Callable
from typing import Tuple
from typing import List
from typing import Union
from util_types import types_of_loco
from network_model.model_base import tempload
from network_model.distillation.distillation_model_builder import DistllationModelIncubator
from keras.optimizers import Optimizer, SGD
from network_model.build_model import builder, builder_with_merge
from model_merger.merge_model import ModelMerger
from keras.layers import Concatenate
from keras.callbacks import Callback
from model_merger.siamese import SiameseBuilder


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
                  optimizer: Optimizer = SGD()) -> ModelBuilder:
    """
    モデル生成をする関数を返す
    交差検証をかける際のラッパーとして使う
    :param img_size:
    :param channels:
    :param model_name:
    :param optimizer:
    :return:
    """
    if model_name == "tempload":
        return lambda load_path: tempload.builder(load_path, optimizer)
    return lambda class_num: builder(class_num, img_size, channels, optimizer, model_name)


def build_with_merge_wrapper(base_model_num: int,
                             img_size: types_of_loco.input_img_size = 28,
                             channels: int = 3,
                             model_name: str = "model1",
                             optimizer: Optimizer = SGD(),
                             model_merger: ModelMerger = ModelMerger(Concatenate(),
                                                                     metrics=['accuracy'])) -> ModelBuilder:
    return lambda class_num: builder_with_merge(base_model_num,
                                                model_merger,
                                                class_num,
                                                img_size,
                                                channels,
                                                optimizer,
                                                model_name)


def build_siamese(q: float,
                  img_size: types_of_loco.input_img_size = 28,
                  channels: int = 3,
                  model_name: str = "model1",
                  optimizer: Optimizer = SGD(),
                  save_best_only=True,
                  save_weights_only=False,
                  save_base_filepath: str = None):
    def build(class_num: int):
        base_model = builder(class_num, img_size, channels, optimizer, model_name)
        shame_builder = SiameseBuilder(base_model)
        return shame_builder.build_shame_trainer_for_classifivation(q,
                                                                    optimizer,
                                                                    save_base_filepath,
                                                                    save_best_only,
                                                                    save_weights_only
                                                                    )
    return build
