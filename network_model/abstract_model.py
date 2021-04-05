import keras.engine.training
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

ModelPreProcessor = Optional[Callable[[keras.engine.training.Model],  keras.engine.training.Model]]


class AbstractModel(ABC):
    def __init__(self,
                 input_shape: Tuple[Optional[int], int, int, int],
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 will_save_h5: bool = True,
                 preprocess_for_model: ModelPreProcessor = None):
        """

         :param input_shape: モデルの入力層の形を表すタプル
         :param class_set: クラスの元となったリスト
         :param callbacks: モデルに渡すコールバック関数
         :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
         :param will_save_h5: 途中モデル読み込み時に旧式のh5ファイルで保存するかどうか　デフォルトだと保存する
         :param preprocess_for_model: モデル学習前にモデルに対してする処理
         """
        self.__class_set = class_set
        self.__input_shape = input_shape
        self.__history = None
        self.__callbacks = callbacks
        self.__monitor = monitor
        self.__will_save_h5 = will_save_h5
        self.__preprocess_for_model = preprocess_for_model

    @property
    @abstractmethod
    def model(self) -> keras.engine.training.Model:
        pass

    @property
    def will_save_h5(self):
        return self.__will_save_h5

    @property
    def will_record_best_model(self):
        return self.__callbacks == ""

    @property
    def class_set(self):
        return self.__class_set

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def preprocess_for_model(self) -> ModelPreProcessor:
        return self.__preprocess_for_model

    def get_callbacks(self, temp_best_path: str, save_weights_only: bool = False):
        if self.will_record_best_model is None or temp_best_path == "":
            return self.__callbacks
        best_model_recorder = keras.callbacks.ModelCheckpoint(temp_best_path,
                                                              monitor=self.__monitor,
                                                              save_best_only=True,
                                                              save_weights_only=save_weights_only)
        return self.__callbacks if self.__callbacks is None else self.__callbacks + [best_model_recorder]

    def predict(self, data: np.ndarray) -> Tuple[np.array, np.array]:
        """
        モデルの適合度から該当するクラスを算出する
        :param data: 算出対象となるデータ
        :return: 判定したインデックスと形式名
        """
        result_set = np.array([np.argmax(result) for result in self.model.predict(data)])
        class_name_set = np.array([self.__class_set[index] for index in result_set])
        return result_set, class_name_set

    def predict_top_n(self, data: np.ndarray, top_num: int = 5) -> List[Tuple[np.array, np.array, np.array]]:
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

    def get_predicted_upper(self, predicted_result: np.ndarray, top_num: int = 5) -> Tuple[
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
        file_name = model_name + ".h5" if self.will_save_h5 else model_name
        self.model.save(os.path.join(result_path, file_name))

    def record_conf_json(self,
                         result_dir_name: str,
                         dir_path: str = os.path.join(os.getcwd(), "result"),
                         normalize_type: Optional[dl.NormalizeType] = None,
                         model_name: str = "model"):
        print("start record conf")
        result_path = build_record_path(result_dir_name, dir_path)
        write_set = {"class_set": self.class_set, "input_shape": self.input_shape}
        write_dic = {model_name: write_set}
        json_path = os.path.join(result_path, "model_conf.json")
        with open(json_path, 'w', encoding='utf8') as fw:
            json.dump(write_dic, fw, ensure_ascii=False)

    def record(self,
               result_dir_name: str,
               dir_path: str = os.path.join(os.getcwd(), "result"),
               model_name: str = "model",
               normalize_type: Optional[dl.NormalizeType] = None):
        now_result_dir_name = result_dir_name + datetime.now().strftime("%Y%m%d%H%M%S")
        self.record_model(now_result_dir_name, dir_path, model_name)
        self.record_conf_json(now_result_dir_name, dir_path, normalize_type, model_name)

    def run_preprocess_model(self, model: keras.engine.training.Model) -> keras.engine.training.Model:
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

