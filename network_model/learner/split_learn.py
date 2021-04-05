import os

from typing import Tuple, List
from typing import Optional
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator
from network_model import model as md
from network_model.model_builder import ModelBuilder
from DataIO.data_loader import NormalizeType
from network_model.learner.abs_split_learner import AbsModelLearner
from network_model.abstract_model import ModelPreProcessor


class ModelLearner(AbsModelLearner):

    def __init__(self,
                 model_builder: ModelBuilder,
                 train_image_generator: ImageDataGenerator,
                 test_image_generator: ImageDataGenerator,
                 class_list: List[str],
                 normalize_type: NormalizeType = NormalizeType.Div255,
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 train_dir_name: str = "train",
                 validation_name: str = "validation",
                 will_save_h5: bool = True,
                 preprocess_for_model: ModelPreProcessor = None):
        """

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
        :param preprocess_for_model: メインの学習前にモデルに対して行う前処理
        """

        super().__init__(model_builder,
                         train_image_generator,
                         test_image_generator,
                         class_list,
                         normalize_type,
                         callbacks,
                         image_size,
                         train_dir_name,
                         validation_name,
                         will_save_h5,
                         preprocess_for_model)

    def build_model(self,
                    tmp_model_path: str = None,
                    monitor: str = "") -> md.ModelForManyData:
        if tmp_model_path is None:
            return md.ModelForManyData(self.model_builder(self.class_num),
                                       self.class_list,
                                       callbacks=self.callbacks,
                                       monitor=monitor,
                                       will_save_h5=self.will_save_h5,
                                       preprocess_for_model=self.preprocess_for_model)

        return md.ModelForManyData(self.model_builder(tmp_model_path),
                                   self.class_list,
                                   callbacks=self.callbacks,
                                   monitor=monitor,
                                   will_save_h5=self.will_save_h5,
                                   preprocess_for_model=self.preprocess_for_model)

    def train_with_validation_from_model(self,
                                         model: md.ModelForManyData,
                                         result_dir_path: str,
                                         train_dir: str,
                                         validation_dir: str,
                                         batch_size=32,
                                         epoch_num: int = 20,
                                         result_name: str = "result",
                                         model_name: str = "model",
                                         save_weights_only: bool = False) -> md.ModelForManyData:
        return super().train_with_validation_from_model(model,
                                                        result_dir_path,
                                                        train_dir,
                                                        validation_dir,
                                                        batch_size,
                                                        epoch_num,
                                                        result_name,
                                                        model_name,
                                                        save_weights_only)

    def train_with_validation(self,
                              dataset_root_dir: str,
                              result_dir_path: str,
                              batch_size=32,
                              epoch_num: int = 20,
                              result_name: str = "result",
                              model_name: str = "model",
                              tmp_model_path: str = None,
                              monitor: str = "",
                              save_weights_only: bool = False) -> md.ModelForManyData:
        """
        検証用データがある場合の学習
        :param dataset_root_dir: データが格納されたディレクトリ
        :param result_dir_path: モデルを出力するディレクトリ
        :param batch_size: 学習する際のバッチサイズ
        :param epoch_num: 学習する際のエポック数
        :param result_name: 出力する結果名
        :param model_name: モデル名
        :param tmp_model_path: 学習済みのh5ファイルからモデルを読み込んで学習する際のh5ファイルのパス
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param save_weights_only:
        :return: 学習済みモデル
        """
        model_val = self.build_model(tmp_model_path, monitor)
        train_dir, validation_dir = self.build_train_validation_dir_paths(dataset_root_dir)
        return self.train_with_validation_from_model(model_val,
                                                     result_dir_path,
                                                     train_dir,
                                                     validation_dir,
                                                     batch_size,
                                                     epoch_num,
                                                     result_name,
                                                     model_name,
                                                     save_weights_only)


