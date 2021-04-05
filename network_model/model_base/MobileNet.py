from keras.applications import MobileNet
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras import optimizers
import keras.engine.training
from typing import Union
from typing import Tuple
from util_types import types_of_loco


def builder(
            class_num: int,
            img_size: types_of_loco.input_img_size = 28,
            channels: int = 3,
            kernel_size: Union[int, Tuple[int, int]] = 3
            ) -> keras.engine.training.Model:
    """
    MobileNet
    :param class_num : 出力するクラスの数
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param kernel_size: 本来はいらないけどインターフェース統一のために残している
    :return: 実行可能なモデル
    """
    mobile_net = MobileNet(include_top=False, input_shape=(img_size, img_size, channels))
    h = Flatten()(mobile_net.output)
    model_output = Dense(class_num, activation="softmax")(h)
    model = Model(mobile_net.input, model_output)

    # モデルの概要を表示
    model.summary()

    # モデルをコンパイル
    model.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(), metrics=['accuracy'])

    return model

