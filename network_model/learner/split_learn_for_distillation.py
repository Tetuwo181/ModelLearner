import os

from typing import Tuple, List
from typing import Optional
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator
from network_model.model_builder import ModelBuilder
from DataIO.data_loader import NormalizeType
from network_model.model_for_distillation import ModelForDistillation
from network_model.distillation.distillation_model_builder import DistllationModelIncubator
from network_model.learner.abs_split_learner import AbsModelLearner
from network_model.distillation.flow_wrapper import FlowForDistillation


class ModelLearnerForDistillation(AbsModelLearner):

    def __init__(self,
                 student_model_name: str,
                 model_builder: ModelBuilder,
                 train_image_generator: ImageDataGenerator,
                 test_image_generator: ImageDataGenerator,
                 class_list: List[str],
                 normalize_type: NormalizeType = NormalizeType.Div255,
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 train_dir_name: str = "train",
                 validation_name: str = "validation",
                 will_save_h5: bool = True):
        """

        :param student_model_name:
        :param model_builder: モデル生成器
        :param train_image_generator: 教師データ生成ジェネレータ
        :param test_image_generator: テストデータ生成ジェネレータ
        :param class_list: 学習対象となるクラス
        :param normalize_type: 正規化の手法
        :param callbacks: kerasで学習させる際のコールバック関数
        :param image_size: 画像の入力サイズ
        :param train_dir_name: 検証する際の教師データのディレクトリ名
        :param validation_name: 検証する際のテストデータのディレクトリ名
        :param will_save_h5: 途中モデル読み込み時に旧式のh5ファイルで保存するかどうか　デフォルトだと保存する
        """

        self.__student_model_name = student_model_name
        super().__init__(model_builder,
                         train_image_generator,
                         test_image_generator,
                         class_list,
                         normalize_type,
                         callbacks,
                         image_size,
                         train_dir_name,
                         validation_name,
                         will_save_h5)

    @property
    def student_model_name(self):
        return self.__student_model_name

    @property
    def model_builder(self) -> DistllationModelIncubator:
        return super().model_builder

    def build_model(self, tmp_model_path: str = None, monitor: str = "") -> ModelForDistillation:
        model_path = self.student_model_name if tmp_model_path is None else tmp_model_path
        return self.model_builder(model_path, self.class_list)

    @FlowForDistillation
    def build_train_generator(self, batch_size, train_dir: str) -> FlowForDistillation:
        return super().build_train_generator(batch_size, train_dir)

    @FlowForDistillation
    def build_test_generator(self, batch_size, test_data_dir: str) -> FlowForDistillation:
        return super().build_test_generator(batch_size, test_data_dir)

    def train_with_validation_from_model(self,
                                         model: ModelForDistillation,
                                         result_dir_path: str,
                                         train_dir: str,
                                         validation_dir: str,
                                         batch_size=32,
                                         epoch_num: int = 20,
                                         result_name: str = "result",
                                         model_name: str = "model",
                                         save_weights_only: bool = False) -> ModelForDistillation:
        return super().train_with_validation_from_model(model,
                                                        result_dir_path,
                                                        train_dir,
                                                        validation_dir,
                                                        batch_size,
                                                        epoch_num,
                                                        result_name,
                                                        model_name,
                                                        save_weights_only)
