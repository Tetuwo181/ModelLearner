from keras.applications import ResNet50
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras import optimizers
import keras.engine.training
from typing import Union
from typing import Tuple
from util_types import types_of_loco
from keras import optimizers


def builder(
            class_num: int,
            img_size: types_of_loco.input_img_size = 28,
            channels: int = 3,
            optimizer=optimizers.SGD()
            ) -> keras.engine.training.Model:
    """
    ResNet50
    こちらはImageNetからの転移学習
    :param class_num : 出力するクラスの数
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param optimizer:
    :return: 実行可能なモデル
    """
    resnet = ResNet50(include_top=False, input_shape=(img_size, img_size, channels))
    h = Flatten()(resnet.output)
    model_output = Dense(class_num, activation="softmax")(h)
    model = Model(resnet.input, model_output)

    # モデルの概要を表示
    model.summary()

    # モデルをコンパイル
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

