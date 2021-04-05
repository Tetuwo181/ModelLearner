# -*- coding: utf-8 -*-
import keras.engine.training
from util_types import types_of_loco
import importlib
from keras.optimizers import Optimizer, SGD


def builder(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        optimizer: Optimizer = SGD(),
        model_name: str = "model1",
) -> keras.engine.training.Model:
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
