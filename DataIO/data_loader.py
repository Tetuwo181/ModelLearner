# -*- coding: utf-8 -*-
from tensorflow.keras.utils import to_categorical
import os
import cv2
import numpy as np
from typing import List
from util_types import two_dim
from typing import Optional
from numba import jit
from enum import Enum

img_size, size_converter = two_dim.init_pair_type(int)


class NormalizeType(Enum):
    NotNormalize = 0
    Div255 = 1
    Div127_5 = 2


def is_image(file_path:str) -> bool:
    """
    拡張子から画像ファイルかどうか判定
    :param file_path:
    :return:
    """
    root, ext = os.path.splitext(file_path)
    return (ext == ".jpg") or (ext == ".jpeg") or (ext == ".JPG") or (ext == ".JPEG") or (ext == ".jfif")


def count_data_num_in_dir(root_dir: str):
    path_set, label_set, class_names, class_num = load_dataset_path(root_dir)
    return len(path_set)


def load_dataset_path(root_dir: str):
    """
    画像データのパスを読み込む
    :param root_dir: 画像データの格納されているルートディレクトリ。直下に存在するディレクトリ名が各画像のクラス名に
    :return: 画像データのパスの配列とラベルの配列とクラスの総数のタプル
    """
    class_names = os.listdir(root_dir)
    print("all classes", class_names)
    encoder = label_encoder(class_names)
    result_path_set = []
    result_label_set = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir,  class_name)
        class_path_set = [os.path.join(class_dir,  data_name) for data_name in os.listdir(class_dir)
                          if is_image(os.path.join(class_dir, data_name))]
        label_converted = [encoder(class_name) for index in range(len(class_path_set))]
        result_path_set.extend(class_path_set)
        result_label_set.extend(label_converted)
        print("class", class_name, "loaded data_num", len(class_path_set))
    return np.array(result_path_set), np.array(result_label_set), class_names, len(class_names)


def load_dataset(root_dir: str,
                 normalize_type: NormalizeType = NormalizeType.Div255,
                 img_resize_val: Optional[img_size] = None,
                 color: str = "RGB"):
    """
    画像データを読み込む
    :param root_dir: 画像データの格納されているルートディレクトリ。直下に存在するディレクトリ名が各画像のクラス名に
    :param normalize_type: どのように正規化するか
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :param color: グレースケールかカラーで読み込むか　デフォルトではカラー(RGB)
    :return: numpy形式の画像データの配列とラベルの配列とクラスの総数のタプル
    """
    class_names = os.listdir(root_dir)
    print("all classes", class_names)
    encoder = label_encoder(class_names)
    result_img_set = []
    result_label_set = []
    for class_name in class_names:
        got_data = load_data_in_dir(os.path.join(root_dir, class_name), normalize_type, img_resize_val, color)
        label_converted = [encoder(class_name) for index in range(len(got_data))]
        result_img_set.extend(got_data)
        result_label_set.extend(label_converted)
        print("class", class_name, "loaded data_num", len(got_data))
    return np.array(result_img_set), np.array(result_label_set), class_names, len(class_names)


def load_data_in_dir(dir_path: str,
                     normalize_type: NormalizeType = NormalizeType.Div255,
                     img_resize_val: Optional[img_size] = None,
                     color: str = "RGB"):
    """
    指定したディレクトリに存在するデータを読み込む
    :param dir_path: 画像データの格納されているディレクトリ。
    :param normalize_type: どのように正規化するか
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :param color: グレースケールかカラーで読み込むか　デフォルトではカラー(RGB)
    :return: numpy形式の画像データの配列
    """
    print("load", dir_path)
    img_path_set = [os.path.join(dir_path,  data_name) for data_name in os.listdir(dir_path)]
    img_set = [load_img(img_path, img_resize_val, color) for img_path in img_path_set]
    if normalize_type == NormalizeType.NotNormalize:
        return np.array(img_set)
    return normalise_img_set(np.array(img_set), normalize_type)


@jit
def load_img(img_path: str, img_resize_val: Optional[img_size] = None,  color: str = "RGB"):
    """
    指定されたパスの画像ファイルを読み込む
    :param img_path: 画像ファイル
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :param color: グレースケールかカラーで読み込むか　デフォルトではカラー(RGB)
    :return:
    """
    raw_img = cv2.imread(img_path)
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) if color == "RGB" else cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    if img_resize_val is None:
        return img
    resize_val = size_converter(img_resize_val)
    return cv2.resize(img, resize_val)


def label_encoder(class_set: List[str]):
    """
    ベースとなるクラスのリストを与えてエンコードする関数を返す
    :param class_set: ベースとなるクラスのリスト
    :return: keras形式でエンコードされたラベルのインデックスを返す関数
    """
    @jit
    def encode(base_class: str):
        """
        クラスとなる文字列を与えてkeras形式でエンコードされたクラスのインデックスを返す
        :param base_class: ベースとなるクラス名
        :return: エンコードされたクラス名
        """
        label_encoded = class_set.index(base_class)
        return to_categorical(label_encoded, len(class_set))
    return encode


@jit
def normalise_img(img: np.ndarray, normalize_type: NormalizeType = NormalizeType.Div255) -> np.ndarray:
    """
    画像を正規化
    :param img: 正規化する対象
    :param normalize_type: どのように正規化するか
    :return: 正規化後の配列
    """
    if NormalizeType(normalize_type) is NormalizeType.Div127_5:
        return (img.astype(np.float32) - 127.5) / 127.5
    if NormalizeType(normalize_type) is NormalizeType.Div255:
        return (img.astype(np.float32)) / 255.0
    return img.astype(np.float32)
    

def normalise_img_set(img_set: np.ndarray, normalize_type: NormalizeType = NormalizeType.Div255) -> np.ndarray:
    """
    画像を入力するために正規化
    :param img_set: ベースになるデータ
    :param normalize_type: どのように正規化するか
    :return: 正規化後のデータ
    """
    return np.array([normalise_img(img, normalize_type) for img in img_set])


@jit
def sampling_real_data_set(batch_num: int, img_set: np.ndarray) -> np.ndarray:
    """
    実際のデータセットからbatch_num分だけデータを復元抽出する
    :param batch_num: データを抽出する数
    :param img_set: 元となるデータセット
    :return: 抽出されたデータセット
     """
    chosen_id_set = np.random.randint(0, img_set.shape[0], batch_num)
    return img_set[chosen_id_set]
