import keras.callbacks
from typing import List
from typing import Optional
import os
from datetime import datetime
from DataIO import data_loader as dl
from network_model.distillation.flow_wrapper import FlowForDistillation
from network_model.wrapper.abstract_model import AbstractModel, build_record_path


class ModelForDistillation(AbstractModel):
    def __init__(self,
                 train_model: keras.engine.training.Model,
                 student_model: keras.engine.training.Model,
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 will_save_h5: bool = True):
        self.__train_model = train_model
        self.__student_model = student_model
        super().__init__(train_model.input.shape.as_list(),
                         class_set,
                         callbacks,
                         monitor,
                         will_save_h5)

    @property
    def model(self):
        return self.__student_model

    def fit_generator(self,
                      image_generator: FlowForDistillation,
                      epochs: int,
                      validation_data: Optional[FlowForDistillation] = None,
                      steps_per_epoch: Optional[int] = None,
                      validation_steps: Optional[int] = None,
                      temp_best_path: str = "",
                      save_weights_only: bool = False):
        """
        モデルの適合度を算出する
        :param image_generator: ファイルパスから学習データを生成する生成器
        :param epochs: エポック数
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプル
        :param steps_per_epoch:
        :param validation_steps:
        :param temp_best_path:
        :param save_weights_only:
        :return:
        """
        callbacks = self.get_callbacks(temp_best_path, save_weights_only)
        print("fit builder")
        if validation_data is None:
            self.__history = self.__student_model.fit_generator(image_generator,
                                                                steps_per_epoch=steps_per_epoch,
                                                                epochs=epochs,
                                                                callbacks=callbacks)
        else:
            print('epochs', epochs)
            self.__history = self.__student_model.fit_generator(image_generator,
                                                                steps_per_epoch=steps_per_epoch,
                                                                validation_steps=validation_steps,
                                                                epochs=epochs,
                                                                validation_data=validation_data,
                                                                callbacks=callbacks)
        return self

    def test(self,
             image_generator: FlowForDistillation,
             epochs: int,
             validation_data: Optional[FlowForDistillation] = None,
             normalize_type: dl.NormalizeType = dl.NormalizeType.Div255,
             result_dir_name: str = None,
             dir_path: str = None,
             model_name: str = None,
             steps_per_epoch: Optional[int] = None,
             validation_steps: Optional[int] = None,
             save_weights_only: bool = False
             ):
        """
        指定したデータセットに対しての正答率を算出する
        :param image_generator: ファイルパスから学習データを生成する生成器
        :param epochs: エポック数
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプルもしくはimage_generatorと同じ形式
        :param epochs: エポック数
        :param normalize_type: どのように正規化するか
        :param result_dir_name: 記録するためのファイル名のベース
        :param dir_path: 記録するディレクトリ デフォルトではカレントディレクトリ直下にresultディレクトリを作成する
        :param model_name: モデル名　デフォルトではmodel
        :param steps_per_epoch: 記録後モデルを削除するかどうか
        :param validation_steps: 記録後モデルを削除するかどうか
        :param save_weights_only:
        :return:学習用データの正答率とテスト用データの正答率のタプル
        """
        write_dir_path = build_record_path(result_dir_name, dir_path)
        save_tmp_name = model_name + "_best.h5" if self.will_save_h5 else model_name + "_best"
        self.fit_generator(image_generator,
                           epochs,
                           validation_data,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           temp_best_path=os.path.join(write_dir_path, save_tmp_name),
                           save_weights_only=save_weights_only)
        now_result_dir_name = result_dir_name + datetime.now().strftime("%Y%m%d%H%M%S")
        self.record_model(now_result_dir_name, dir_path, model_name)
        self.record_conf_json(now_result_dir_name, dir_path, normalize_type, model_name)


