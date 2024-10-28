# -*- coding: utf-8 -*-
from __future__ import annotations
from tensorflow.keras import Model
from util_types import types_of_loco
import importlib
from tensorflow.keras.optimizers import Optimizer, SGD
from model_merger.keras.merge_model import ModelMerger, TrainableModelIndex
from typing import List, Tuple, Union, Optional
from keras.layers import Concatenate


def builder(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        optimizer: Optimizer = SGD(),
        model_name: str = "model1",
) -> Model:
    """
    モデルを作成する
    :param class_num : 出力するクラス数
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param optimizer: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :param model_name: インポートするモデルの名前。models_base以下のディレクトリにモデル生成器を置く
    :return: discriminator部のモデル
    """
    model_module = importlib.import_module("network_model.model_base."+model_name)
    return model_module.builder(class_num, img_size, channels, optimizer)


def builder_pt(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        model_name: str = "model1",
) -> Model:
    """
    モデルを作成する
    :param class_num : 出力するクラス数
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param model_name: インポートするモデルの名前。models_base以下のディレクトリにモデル生成器を置く
    :return: discriminator部のモデル
    """
    model_module = importlib.import_module("network_model.model_base."+model_name)
    return model_module.builder(class_num, img_size)


def builder_with_merge(
        base_model_num: int,
        model_merger: ModelMerger,
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        optimizer: Optimizer = SGD(),
        model_name: str = "model1",
) -> Model:
    models = [builder(class_num, img_size, channels, optimizer,  model_name) for _ in range(base_model_num)]
    return model_merger.merge_models_separately_input(models, class_num)


def build_by_h5files(
                     h5_paths: list[str | tuple[str, str]],
                     trainable_model: TrainableModelIndex | list[TrainableModelIndex] = True,
                     output_num: Optional[int] = None,
                     middle_layer_neuro_nums: list[tuple[int, str]] | None = None,
                     merge_per_model_name: str = 'model',
                     model_merger: ModelMerger = ModelMerger(Concatenate())
) -> Model:
    return model_merger.merge_models_from_model_files(h5_paths,
                                                      trainable_model,
                                                      output_num,
                                                      middle_layer_neuro_nums,
                                                      merge_per_model_name)
