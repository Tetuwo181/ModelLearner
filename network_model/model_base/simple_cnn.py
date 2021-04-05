# ネットワーク構築に必要なものをインポート
from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
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
    シンプルなCNNを構築
    今回は畳み込み層→プーリング層→1次元に変換→出力層となる
    :param class_num : 出力するクラスの数
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :return: 実行可能なモデル
    """

    # 入力部の設定
    img_shape = (img_size, img_size, channels)

    model = Sequential()

    # 畳み込み層を追加
    model.add(Conv2D(img_size * 2, kernel_size=kernel_size, activation='relu', input_shape=img_shape))

    # プーリング層を追加
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト率も設定
    model.add(Dropout(0.25))

    # 入力を1次元に変換
    model.add(Flatten())
    model.add(Dense(class_num, activation='softmax'))

    # モデルの概要を表示
    model.summary()

    # モデルをコンパイル
    model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(), metrics=['accuracy'])

    return model
