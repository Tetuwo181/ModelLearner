from __future__ import annotations
from tensorflow.keras import Model
import os

from typing import  Callable, Union
from typing import Optional
import keras.callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from network_model.wrapper.keras import many_data as md
from network_model.model_builder import ModelBuilder
from DataIO.data_loader import NormalizeType
from network_model.learner.abs_split_learner import AbsModelLearner
from network_model.builder.pytorch_builder import PytorchModelBuilder


class ModelLearner(AbsModelLearner):

    def __init__(self,
                 model_builder: ModelBuilder | PytorchModelBuilder,
                 train_image_generator: ImageDataGenerator,
                 test_image_generator: ImageDataGenerator,
                 class_list: list[str],
                 normalize_type: NormalizeType = NormalizeType.Div255,
                 callbacks: list[keras.callbacks.Callback] | None = None,
                 image_size: tuple[int, int] = (224, 224),
                 train_dir_name: str = "train",
                 validation_name: str = "validation",
                 will_save_h5: bool = True,
                 preprocess_for_model=None,
                 after_learned_process: Optional[Callable[[None], None]] = None,
                 class_mode: Optional[str] = None,
                 class_num: Optional[int] = None):
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
        :param after_learned_process: モデル学習後の後始末
        :param class_mode: flow_from_directoryのクラスモード
        :param class_num: 出力するクラス数　デフォルトではクラスのリスト長と同じになる
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
                         preprocess_for_model,
                         after_learned_process,
                         class_mode,
                         class_num)

    def build_model_from_result(self,
                                build_result,
                                model_dir_path: str,
                                result_name: str,
                                monitor: str = ""):
        if self.is_torch:
            return build_result(self.class_list,
                                self.callbacks,
                                monitor,
                                self.preprocess_for_model,
                                self.after_learned_process)

        if type(build_result) is not tuple:
            model = build_result
            return md.ModelForManyData(model,
                                       self.class_list,
                                       callbacks=self.callbacks,
                                       monitor=monitor,
                                       will_save_h5=self.will_save_h5,
                                       preprocess_for_model=self.preprocess_for_model,
                                       after_learned_process=self.after_learned_process)
        model = build_result[0]
        if type(build_result[1]) is list:
            callbacks = self.callbacks + build_result[1]
            return md.ModelForManyData(model,
                                       self.class_list,
                                       callbacks=callbacks,
                                       monitor=monitor,
                                       will_save_h5=self.will_save_h5,
                                       preprocess_for_model=self.preprocess_for_model,
                                       after_learned_process=self.after_learned_process)
        base_model_name = result_name + "original.h5" if self.will_save_h5 else result_name + "original"
        callbacks = self.callbacks + [build_result[1](os.path.join(model_dir_path,
                                                                   result_name,
                                                                   base_model_name))]
        return md.ModelForManyData(model,
                                   self.class_list,
                                   callbacks=callbacks,
                                   monitor=monitor,
                                   will_save_h5=self.will_save_h5,
                                   preprocess_for_model=self.preprocess_for_model,
                                   after_learned_process=self.after_learned_process)

    def build_model(self,
                    model_dir_path: str,
                    result_name: str,
                    tmp_model_path: str = None,
                    monitor: str = "") -> md.ModelForManyData:
        model_builder_input = self.class_num if tmp_model_path is None else tmp_model_path
        build_result = self.model_builder(model_builder_input)
        return self.build_model_from_result(build_result,
                                            model_dir_path,
                                            result_name,
                                            monitor)

    def train_with_validation_from_model(self,
                                         model: md.ModelForManyData,
                                         result_dir_path: str,
                                         train_dir: str,
                                         validation_dir: str,
                                         batch_size=32,
                                         epoch_num: int = 20,
                                         result_name: str = "result",
                                         model_name: str = "model",
                                         save_weights_only: bool = False,
                                         will_use_multi_inputs_per_one_image: bool = False,
                                         data_preprocess=None) -> md.ModelForManyData:
        return super().train_with_validation_from_model(model,
                                                        result_dir_path,
                                                        train_dir,
                                                        validation_dir,
                                                        batch_size,
                                                        epoch_num,
                                                        result_name,
                                                        model_name,
                                                        save_weights_only,
                                                        will_use_multi_inputs_per_one_image,
                                                        data_preprocess)

    def train_with_validation(self,
                              dataset_root_dir: str,
                              result_dir_path: str,
                              batch_size=32,
                              epoch_num: int = 20,
                              result_name: str = "result",
                              model_name: str = "model",
                              tmp_model_path: str = None,
                              monitor: str = "",
                              save_weights_only: bool = False,
                              will_use_multi_inputs_per_one_image: bool = False,
                              data_preprocess=None) -> md.ModelForManyData:
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
        :param will_use_multi_inputs_per_one_image:
        :param data_preprocess:
        :return: 学習済みモデル
        """
        model_val = self.build_model(result_dir_path, result_name, tmp_model_path, monitor)
        train_dir, validation_dir = self.build_train_validation_dir_paths(dataset_root_dir)
        return self.train_with_validation_from_model(model_val,
                                                     result_dir_path,
                                                     train_dir,
                                                     validation_dir,
                                                     batch_size,
                                                     epoch_num,
                                                     result_name,
                                                     model_name,
                                                     save_weights_only,
                                                     will_use_multi_inputs_per_one_image,
                                                     data_preprocess)


