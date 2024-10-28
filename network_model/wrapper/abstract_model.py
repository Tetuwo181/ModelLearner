from __future__ import annotations
from tensorflow.keras import Model
import keras.callbacks
import numpy as np
from typing import List
from typing import Tuple
from typing import Optional
from typing import Callable
import os
from datetime import datetime
import json
from DataIO import data_loader as dl
from abc import ABC, abstractmethod
from util.keras_version import is_new_keras


class AbstractModel(ABC):
    def __init__(self,
                 class_set: list[str],
                 callbacks: list[keras.callbacks.Callback] | None = None,
                 monitor: str = "",
                 preprocess_for_model=None,
                 after_learned_process: Callable[[None], None] | None = None):
        """

         :param class_set: クラスの元となったリスト
         :param callbacks: モデルに渡すコールバック関数
         :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
         :param preprocess_for_model: モデル学習前にモデルに対してする処理
         :param after_learned_process: モデル学習後の後始末
         """
        self.__class_set = class_set
        self.__history = None
        self.__callbacks = callbacks
        self.__monitor = monitor
        self.__preprocess_for_model = preprocess_for_model
        self.__after_learned_process = after_learned_process
        if is_new_keras() is False and self.__monitor == "val_accuracy":
            self.__monitor = "val_acc"

    @property
    @abstractmethod
    def model(self) -> keras.engine.training.Model:
        pass

    @abstractmethod
    def build_model_checkpoint(self, temp_best_path, save_weights_only):
        pass

    @property
    def will_record_best_model(self):
        return self.__callbacks == ""

    @property
    def class_set(self):
        return self.__class_set

    @property
    def preprocess_for_model(self):
        return self.__preprocess_for_model

    @property
    def monitor(self):
        return self.__monitor

    def after_learned_process(self):
        if self.__after_learned_process is None:
            return
        return self.__after_learned_process()

    def get_callbacks(self, temp_best_path: str, save_weights_only: bool = False):
        if self.will_record_best_model is None or temp_best_path == "":
            return self.__callbacks
        best_model_recorder = self.build_model_checkpoint(temp_best_path, save_weights_only)
        return self.__callbacks if self.__callbacks is None else self.__callbacks + [best_model_recorder]

    def predict(self, data: np.ndarray) -> tuple[np.array, np.array]:
        """
        モデルの適合度から該当するクラスを算出する
        :param data: 算出対象となるデータ
        :return: 判定したインデックスと形式名
        """
        result_set = np.array([np.argmax(result) for result in self.model.predict(data)])
        class_name_set = np.array([self.__class_set[index] for index in result_set])
        return result_set, class_name_set

    def predict_top_n(self, data: np.ndarray, top_num: int = 5) -> list[tuple[np.array, np.array, np.array]]:
        """
        適合度が高い順に車両形式を取得する
        :param data: 算出対象となるデータ
        :param top_num: 取得する上位の数値
        :return: 判定したインデックスと形式名と確率のタプルのリスト
        """
        predicted_set = self.model.predict(data)
        return [self.get_predicted_upper(predicted_result, top_num) for predicted_result in predicted_set]

    def calc_succeed_rate(self,
                          data_set: np.ndarray,
                          label_set: np.ndarray, ) -> float:
        """
        指定したデータセットに対しての正答率を算出する
        :param data_set: テストデータ
        :param label_set: 正解のラベル
        :return:
        """
        predicted_index, predicted_name = self.predict(data_set)
        teacher_label_set = np.array([np.argmax(teacher_label) for teacher_label in label_set])
        # 教師データと予測されたデータの差が0でなければ誤判定
        diff = teacher_label_set - predicted_index
        return np.sum(diff == 0) / len(data_set)

    def get_predicted_upper(self, predicted_result: np.ndarray, top_num: int = 5) -> tuple[
        np.array, np.array, np.array]:
        """
        予測した結果の数値からふさわしい形式を指定した上位n個だけ取り出す
        :param predicted_result: 予測結果
        :param top_num:
        :return:
        """
        top_index_set = np.argpartition(-predicted_result, top_num)[:top_num]
        top_value_set = predicted_result[top_index_set]
        top_series_set = np.array([self.class_set[index] for index in top_index_set])
        return top_index_set, top_value_set, top_series_set

    def record_model(self,
                     result_dir_name: str,
                     dir_path: str = os.path.join(os.getcwd(), "result"),
                     model_name: str = "model"):
        print("start record")
        result_path = build_record_path(result_dir_name, dir_path)
        file_name = self.build_model_file_name(model_name)
        self.save_model(os.path.join(result_path, file_name))

    @abstractmethod
    def save_model(self, file_path):
        pass

    def record_conf_json(self,
                         result_dir_name: str,
                         dir_path: str = os.path.join(os.getcwd(), "result"),
                         normalize_type: Optional[dl.NormalizeType] = None,
                         model_name: str = "model"):
        print("start record conf")
        result_path = build_record_path(result_dir_name, dir_path)
        write_set = self.build_write_set()
        write_dic = {model_name: write_set}
        json_path = os.path.join(result_path, "model_conf.json")
        with open(json_path, 'w', encoding='utf8') as fw:
            json.dump(write_dic, fw, ensure_ascii=False)

    @abstractmethod
    def build_model_file_name(self, model_name):
        pass

    @abstractmethod
    def build_write_set(self):
        pass

    @abstractmethod
    def build_best_model_file_name(self, model_name):
        pass

    def record(self,
               result_dir_name: str,
               dir_path: str = os.path.join(os.getcwd(), "result"),
               model_name: str = "model",
               normalize_type: Optional[dl.NormalizeType] = None):
        self.record_model(result_dir_name, dir_path, model_name)
        self.record_conf_json(result_dir_name, dir_path, normalize_type, model_name)

    def run_preprocess_model(self, model: Model) -> Model:
        if self.preprocess_for_model is None:
            return model
        return self.preprocess_for_model(model)


def build_record_path(result_dir_name, dir_path):
    if not os.path.exists(dir_path):
        print('create dir', dir_path)
        os.makedirs(dir_path)
    result_path = os.path.join(dir_path, result_dir_name)
    if not os.path.exists(result_path):
        print('create dir', result_path)
        os.makedirs(result_path)
    print('write to ', result_path)
    return result_path

