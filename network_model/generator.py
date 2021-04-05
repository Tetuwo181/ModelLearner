from typing import List
from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple
from keras.utils import Sequence
import numpy as np
from DataIO.data_loader import load_img
from DataIO.data_loader import normalise_img_set
from DataIO.data_loader import normalise_img
from keras.preprocessing.image import ImageDataGenerator
from DataIO.data_loader import NormalizeType
from util_types import two_dim


img_size, size_converter = two_dim.init_pair_type(int)


class DataLoaderFromPaths(Sequence):
    """
    データを指定したパスから読み込むジェネレータ
    データセットがメモリに乗りきらない場合に使う
    """

    def __init__(self,
                 data_paths: List[str],
                 data_classes: List[str],
                 class_num: int,
                 batch_size: int = 1,
                 img_resize_val: Optional[img_size] = None,
                 color: str = "RGB",
                 normalize_type: NormalizeType = NormalizeType.NotNormalize):
        """
        :param data_paths: データセットのパスのリスト
        :param data_classes: 各データセットのクラス
        :param class_num: クラスの種類の数
        :param batch_size: バッチサイズ
        :param img_resize_val: データ読み込みした後リサイズするかどうか　デフォルトではそのままのサイズで読み込み
        :param color: カラー RGB以外なら白黒扱い
        :param normalize_type: データ正規化のタイプ
        """
        self.__data_paths = data_paths
        self.__data_classes = data_classes
        self.__length = len(data_paths)
        self.__batch_size = batch_size
        self.__img_resize_val = img_resize_val
        self.__color = color
        self.__class_num = class_num
        self.__num_batches_per_epoch = int((self.__length - 1) / batch_size) + 1
        self.__normalize_type = normalize_type
        print("initialized data_loader")

    def __getitem__(self, idx):
        """Get batch data
        :param idx: Index of batch
        :return imgs: numpy array of images
        :return labels: numpy array of label
        """

        start_pos = self.__batch_size * idx
        end_pos = start_pos + self.__batch_size
        if end_pos > self.__length:
            end_pos = self.__length
        item_paths = self.__data_paths[start_pos: end_pos]
        labels = self.__data_classes[start_pos: end_pos]
        image_set = np.array([load_img(path, self.__img_resize_val, self.__color) for path in item_paths])

        return normalise_img_set(image_set, self.__normalize_type), np.array(labels)

    def __len__(self):
        """Batch length"""
        return self.__num_batches_per_epoch


class DataLoaderFromPathsWithDataAugmentation(Sequence):
    """
    データを指定したパスから読み込むジェネレータ
    データセットがメモリに乗りきらない場合に使う
    こちらはデータを水増ししたい場合に使う
    """

    def __init__(self,
                 data_paths: List[str],
                 data_classes: List[str],
                 class_num: int,
                 image_generator: ImageDataGenerator,
                 augmentation_batch_size: int = 8,
                 build_original_data_num: int = 4,
                 img_resize_val: Optional[img_size] = None,
                 color: str = "RGB",
                 normalize_type: NormalizeType = NormalizeType.NotNormalize):
        """

        :param data_paths: データセットのパスのリスト
        :param data_classes: 各データセットのクラス
        :param class_num: クラスの種類の数
        :param image_generator: データの水増しを行うジェネレータ
        :param augmentation_batch_size: 1枚のデータから水増しするデータの枚数
        :param build_original_data_num: 1回の試行あたりでまとめて水増しするデータの数
        :param img_resize_val: データ読み込みした後リサイズするかどうか　デフォルトではそのままのサイズで読み込み
        :param color: カラー RGB以外なら白黒扱い
        :param normalize_type: データ正規化のタイプ
        """
        self.__data_paths = data_paths
        self.__data_classes = data_classes
        self.__original_data_length = len(data_paths)
        self.__length = len(data_paths) * build_original_data_num
        self.__image_generator = image_generator
        self.__augmentation_batch_size = augmentation_batch_size
        self.__build_original_data_num = build_original_data_num
        self.__batch_size = self.__augmentation_batch_size * self.__build_original_data_num
        self.__img_resize_val = img_resize_val
        self.__color = color
        self.__class_num = class_num
        self.__num_batches_per_epoch = ((self.__length - 1) // self.__batch_size) + 1
        self.__normalize_type = normalize_type
        print("initialized data_loader with augument")

    def __getitem__(self, idx):
        """Get batch data
        :param idx: Index of batch
        :return imgs: numpy array of images
        :return labels: numpy array of label
        """

        start_pos = self.__build_original_data_num * idx
        end_pos = start_pos + self.__build_original_data_num
        if end_pos > self.__original_data_length:
            end_pos = self.__original_data_length
        item_paths = self.__data_paths[start_pos: end_pos]
        labels = self.__data_classes[start_pos: end_pos]
        image_set = np.array([load_img(path, self.__img_resize_val, self.__color) for path in item_paths])
        build_base = self.__image_generator.flow(image_set, labels, batch_size=self.__build_original_data_num)
        return self.build_data(build_base)

    def __len__(self):
        """Batch length"""
        return self.__num_batches_per_epoch

    def build_data(self,
                   data_flow,
                   ) -> Tuple[np.ndarray, np.ndarray]:
        result_data = []
        result_class = []
        for batch_num, (data_set, class_set) in enumerate(data_flow):
            result_data.extend(np.array(data_set))
            result_class.extend(np.array(class_set))
            if batch_num >= self.__augmentation_batch_size:
                return normalise_img_set(np.array(result_data), self.__normalize_type), np.array(result_class)


def init_loader_setting(
                        class_num: int,
                        img_resize_val: Optional[img_size] = None,
                        color: str = "RGB",
                        normalize_type: NormalizeType = NormalizeType.NotNormalize
                        ):
    """

     :param class_num: クラスの種類の数
     :param img_resize_val: データ読み込みした後リサイズするかどうか　デフォルトではそのままのサイズで読み込み
     :param color: カラー RGB以外なら白黒扱い
     :param normalize_type: データ正規化のタイプ
    :return:
    """
    def build_data_loader(data_paths: List[str],
                          data_classes: List[str],
                          batch_size: int = 1) -> DataLoaderFromPaths:
        """

        :param data_paths: データセットのパスのリスト
        :param data_classes: 各データセットのクラス
        :param batch_size: バッチサイズ
        :return:
        """
        return DataLoaderFromPaths(data_paths,
                                   data_classes,
                                   class_num,
                                   batch_size,
                                   img_resize_val,
                                   color,
                                   normalize_type
                                   )

    def build_with_data_augmentation(data_paths: List[str],
                                     data_classes: List[str],
                                     image_generator: ImageDataGenerator,
                                     augmentation_batch_size: int = 8,
                                     build_original_data_num: int = 4) -> DataLoaderFromPathsWithDataAugmentation:
        """

        :param data_paths: データセットのパスのリスト
        :param data_classes: 各データセットのクラス
        :param image_generator: データの水増しを行うジェネレータ
        :param augmentation_batch_size: 1枚のデータから水増しするデータの枚数
        :param build_original_data_num: 1回の試行あたりでまとめて水増しするデータの数
        :return:
        """
        return DataLoaderFromPathsWithDataAugmentation(data_paths,
                                                       data_classes,
                                                       class_num,
                                                       image_generator,
                                                       augmentation_batch_size,
                                                       build_original_data_num,
                                                       img_resize_val,
                                                       color,
                                                       normalize_type
                                                       )

    def build(image_generator: Optional[ImageDataGenerator] = None) -> Union[Callable[[List[str], List[str], int],
                                                                                      DataLoaderFromPaths],
                                                                             Callable[[List[str], List[str], int, int],
                                                                             DataLoaderFromPathsWithDataAugmentation]]:
        """

        :param image_generator: データの水増しを行うジェネレータ
        :return:
        """
        def build_with_data_augmentation_wrap(data_paths: List[str],
                                              data_classes: List[str],
                                              augmentation_batch_size: int = 8,
                                              build_original_data_num: int = 4
                                              ) -> DataLoaderFromPathsWithDataAugmentation:
            """
            :param data_paths: データセットのパスのリスト
            :param data_classes: 各データセットのクラス
            :param augmentation_batch_size: 1枚のデータから水増しするデータの枚数
            :param build_original_data_num: 1回の試行あたりでまとめて水増しするデータの数
            :return:
            """
            return build_with_data_augmentation(data_paths,
                                                data_classes,
                                                image_generator,
                                                augmentation_batch_size,
                                                build_original_data_num)
        if image_generator is None:
            return build_data_loader
        return build_with_data_augmentation_wrap

    return build
