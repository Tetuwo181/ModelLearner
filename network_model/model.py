import keras.callbacks
from keras.callbacks import CallbackList, ProgbarLogger, BaseLogger, History
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from network_model.generator import DataLoaderFromPaths
from network_model.generator import DataLoaderFromPathsWithDataAugmentation
from typing import List
from typing import Tuple
from typing import Optional
from typing import Union
from typing import Callable
import os
from datetime import datetime
from DataIO import data_loader as dl
from network_model.abstract_model import AbstractModel, build_record_path, ModelPreProcessor


class Model(AbstractModel):

    def __init__(self,
                 model_base: keras.engine.training.Model,
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 will_save_h5: bool = True,
                 preprocess_for_model: ModelPreProcessor = None,
                 after_learned_process: Optional[Callable[[None], None]] = None):
        """

        :param model_base: kerasで構築したモデル
        :param class_set: クラスの元となったリスト
        :param callbacks: モデルに渡すコールバック関数
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param will_save_h5: 途中モデル読み込み時に旧式のh5ファイルで保存するかどうか　デフォルトだと保存する
        :param preprocess_for_model: モデル学習前にモデルに対してする処理
        :param after_learned_process: モデル学習後の後始末
        """
        self.__model = model_base
        shape = model_base.input[0].shape.as_list() if type(model_base.input) is list else model_base.input.shape.as_list()
        super().__init__(shape,
                         class_set,
                         callbacks,
                         monitor,
                         will_save_h5,
                         preprocess_for_model,
                         after_learned_process)

    @property
    def model(self):
        return self.__model

    def fit(self,
            data: np.ndarray,
            label_set: np.ndarray,
            epochs: int,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            temp_best_path: str = "",
            save_weights_only: bool = False):
        """
        モデルの適合度を算出する
        :param data: 学習に使うデータ
        :param label_set: 教師ラベル
        :param epochs: エポック数
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプル
        :param temp_best_path:
        :param save_weights_only:
        :return:
        """
        callbacks = self.get_callbacks(temp_best_path, save_weights_only)
        self.__model = self.run_preprocess_model(self.__model)
        if validation_data is None:
            self.__model.fit(data, label_set, epochs=epochs, callbacks=callbacks)
        else:
            self.__model.fit(data, label_set, epochs=epochs, validation_data=validation_data, callbacks=callbacks)
        self.after_learned_process()
        return self

    def fit_generator(self,
                      image_generator: ImageDataGenerator,
                      data: np.ndarray,
                      label_set: np.ndarray,
                      epochs: int,
                      generator_batch_size: int = 32,
                      validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      temp_best_path: str = "",
                      save_weights_only: bool = False):
        """
        モデルの適合度を算出する
        generatorを使ってデータを水増しして学習する場合に使用する
        :param image_generator: keras形式でのデータを水増しするジェネレータ
        :param data: 学習に使うデータ
        :param label_set: 教師ラベル
        :param epochs: エポック数
        :param generator_batch_size: ジェネレータのバッチサイズ
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプル
        :param temp_best_path:
        :param save_weights_only:
        :return:
        """
        callbacks = self.get_callbacks(temp_best_path, save_weights_only)
        print("fit generator")
        image_generator.fit(data)
        print("start learning")
        self.__model = self.run_preprocess_model(self.__model)
        if validation_data is None:
            self.__history = self.__model.fit(image_generator.flow(data,
                                                                   label_set,
                                                                   batch_size=generator_batch_size),
                                              steps_per_epoch=len(data) / generator_batch_size,
                                              epochs=epochs,
                                              callbacks=callbacks)
        else:
            self.__history = self.__model.fit(image_generator.flow(data,
                                                                   label_set,
                                                                   batch_size=generator_batch_size),
                                              steps_per_epoch=len(data) / generator_batch_size,
                                              epochs=epochs,
                                              validation_data=validation_data,
                                              callbacks=callbacks)
        self.after_learned_process()
        return self

    def predict(self, data: np.ndarray) -> Tuple[np.array, np.array]:
        """
        モデルの適合度から該当するクラスを算出する
        :param data: 算出対象となるデータ
        :return: 判定したインデックスと形式名
        """
        result_set = np.array([np.argmax(result) for result in self.__model.predict(data)])
        class_name_set = np.array([self.__class_set[index] for index in result_set])
        return result_set, class_name_set

    def test(self,
             train_data_set: np.ndarray,
             train_label_set: np.ndarray,
             test_data_set: np.ndarray,
             test_label_set: np.ndarray,
             epochs: int,
             normalize_type: dl.NormalizeType = dl.NormalizeType.Div255,
             image_generator: ImageDataGenerator = None,
             generator_batch_size: int = 32,
             result_dir_name: str = None,
             dir_path: str = None,
             model_name: str = None,
             save_weights_only: bool = False):
        """
        指定したデータセットに対しての正答率を算出する
        :param train_data_set: 学習に使用したデータ
        :param train_label_set: 学習に使用した正解のラベル
        :param test_data_set: テストデータ
        :param test_label_set: テストのラベル
        :param epochs: エポック数
        :param normalize_type: どのように正規化するか
        :param image_generator: keras形式でのデータを水増しするジェネレータ これを引数で渡さない場合はデータの水増しをしない
        :param generator_batch_size: ジェネレータのバッチサイズ
        :param result_dir_name: 記録するためのファイル名のベース
        :param dir_path: 記録するディレクトリ デフォルトではカレントディレクトリ直下にresultディレクトリを作成する
        :param model_name: モデル名　デフォルトではmodel
        :param save_weights_only:
        :return:学習用データの正答率とテスト用データの正答率のタプル
        """
        save_tmp_name = model_name + "_best.h5" if self.will_save_h5 else model_name + "_best"
        if image_generator is None:
            self.fit(train_data_set,
                     train_label_set,
                     epochs,
                     (test_data_set, test_label_set),
                     temp_best_path=save_tmp_name,
                     save_weights_only=save_weights_only)
        else:
            self.fit_generator(image_generator,
                               train_data_set,
                               train_label_set,
                               epochs,
                               generator_batch_size,
                               (test_data_set, test_label_set),
                               temp_best_path=save_tmp_name,
                               save_weights_only=save_weights_only)
        now_result_dir_name = result_dir_name + datetime.now().strftime("%Y%m%d%H%M%S")
        self.record_model(now_result_dir_name, dir_path, model_name)
        self.record_conf_json(now_result_dir_name, dir_path, normalize_type, model_name)
        train_rate = self.calc_succeed_rate(train_data_set, train_label_set)
        test_rate = self.calc_succeed_rate(test_data_set, test_label_set)
        # 教師データと予測されたデータの差が0でなければ誤判定

        return train_rate, test_rate


class ModelForManyData(AbstractModel):
    """
    メモリに乗りきらない量のデータの学習を行う場合はこちらのクラスを使う
    """

    def __init__(self,
                 model_base: keras.engine.training.Model,
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 will_save_h5: bool = True,
                 preprocess_for_model: ModelPreProcessor = None,
                 after_learned_process: Optional[Callable[[None], None]] = None):
        """

        :param model_base: kerasで構築したモデル
        :param class_set: クラスの元となったリスト
        :param callbacks: モデルに渡すコールバック関数
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param will_save_h5: 途中モデル読み込み時に旧式のh5ファイルで保存するかどうか　デフォルトだと保存する
        :param preprocess_for_model: モデル学習前にモデルに対してする処理
        :param after_learned_process: モデル学習後の後始末
        """
        self.__model = model_base
        self.__model = model_base
        shape = model_base.input[0].shape.as_list() if type(model_base.input) is list else model_base.input.shape.as_list()
        super().__init__(shape,
                         class_set,
                         callbacks,
                         monitor,
                         will_save_h5,
                         preprocess_for_model,
                         after_learned_process)

    @property
    def callbacks_metric(self):
        out_labels = self.model.metrics_names
        return ['val_' + n for n in out_labels]

    @property
    def base_logger(self):
        return BaseLogger(stateful_metrics=self.model.stateful_metric_names)

    @property
    def progbar_logger(self):
        return ProgbarLogger(count_mode='steps', stateful_metrics=self.model.stateful_metric_names)

    def get_callbacks_for_preprocess_input(self, temp_best_path, save_weights_only= False):
        base_callbacks = self.get_callbacks(temp_best_path, save_weights_only)
        if base_callbacks is None or base_callbacks == []:
            return [self.model.history]
        return base_callbacks + [self.model.history]

    def build_callbacks_for_preprocess_input(self,
                                             epochs: int,
                                             temp_best_path,
                                             steps_per_epoch: Optional[int] = None,
                                             save_weights_only=False):
        self.__model.history = History()
        build_callbacks = [self.base_logger, self.progbar_logger]
        raw_callbacks = build_callbacks + self.get_callbacks_for_preprocess_input(temp_best_path, save_weights_only)
        callbacks = CallbackList(raw_callbacks)
        callbacks.set_model(self.model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1,
            'do_validation': False,
            'metrics': self.callbacks_metric,
        })
        return callbacks


    @property
    def model(self):
        return self.__model

    def fit_generator(self,
                      image_generator: Union[DataLoaderFromPathsWithDataAugmentation, DataLoaderFromPaths],
                      epochs: int,
                      validation_data: Union[Optional[Tuple[np.ndarray, np.ndarray]],
                                             DataLoaderFromPathsWithDataAugmentation,
                                             DataLoaderFromPaths] = None,
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
        print("fit generator")
        self.__model = self.run_preprocess_model(self.__model)
        if validation_data is None:
            self.__history = self.__model.fit(image_generator,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=epochs,
                                              callbacks=callbacks)
        else:
            print('epochs', epochs)
            self.__history = self.__model.fit(image_generator,
                                              steps_per_epoch=steps_per_epoch,
                                              validation_steps=validation_steps,
                                              epochs=epochs,
                                              validation_data=validation_data,
                                              callbacks=callbacks)
        self.after_learned_process()
        return self

    def test(self,
             image_generator: Union[DataLoaderFromPathsWithDataAugmentation, DataLoaderFromPaths],
             epochs: int,
             validation_data: Union[Optional[Tuple[np.ndarray, np.ndarray]],
                                    DataLoaderFromPathsWithDataAugmentation,
                                    DataLoaderFromPaths] = None,
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
        self.record_model(result_dir_name, dir_path, model_name)
        self.record_conf_json(result_dir_name, dir_path, normalize_type, model_name)

