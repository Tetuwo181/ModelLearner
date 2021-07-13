import keras.callbacks
from keras.callbacks import CallbackList, ProgbarLogger, BaseLogger, History
from keras.utils.data_utils import Sequence, OrderedEnqueuer
from keras.utils.generic_utils import to_list, unpack_singleton
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
from util.keras_version import is_new_keras


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
            if is_new_keras():
                self.__history = self.__model.fit(image_generator.flow(data,
                                                                       label_set,
                                                                       batch_size=generator_batch_size),
                                                  steps_per_epoch=len(data) / generator_batch_size,
                                                  epochs=epochs,
                                                  callbacks=callbacks)
            else:
                self.__history = self.__model.fit_generator(image_generator.flow(data,
                                                                                 label_set,
                                                                                 batch_size=generator_batch_size),
                                                            steps_per_epoch=len(data) / generator_batch_size,
                                                            epochs=epochs,
                                                            callbacks=callbacks)

        else:
            if is_new_keras():
                self.__history = self.__model.fit(image_generator.flow(data,
                                                                       label_set,
                                                                       batch_size=generator_batch_size),
                                                  steps_per_epoch=len(data) / generator_batch_size,
                                                  epochs=epochs,
                                                  validation_data=validation_data,
                                                  callbacks=callbacks)
            else:
                self.__history = self.__model.fit_generator(image_generator.flow(data,
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
        return BaseLogger(stateful_metrics=self.stateful_metric_names)

    @property
    def progbar_logger(self):
        return ProgbarLogger(count_mode='steps', stateful_metrics=self.stateful_metric_names)

    @property
    def stateful_metric_names(self):
        return ["loss", "accuracy", "val_loss", "val_accuracy"]

    def get_callbacks_for_multi_input(self, temp_best_path, save_weights_only= False):
        base_callbacks = self.get_callbacks(temp_best_path, save_weights_only)
        if base_callbacks is None or base_callbacks == []:
            return [self.model.history]
        return base_callbacks + [self.model.history]

    def build_callbacks_for_multi_input(self,
                                        epochs: int,
                                        temp_best_path,
                                        steps_per_epoch: Optional[int] = None,
                                        validation_data: Union[Optional[Tuple[np.ndarray, np.ndarray]],
                                                               DataLoaderFromPathsWithDataAugmentation,
                                                               DataLoaderFromPaths] = None,
                                        save_weights_only=False):
        """
        一つのデータから複数の入力を使用する場合のコールバックを生成する
        :param epochs: エポック数
        :param temp_best_path:
        :param steps_per_epoch:
        :param validation_data
        :param save_weights_only:
        :return:
        """
        self.__model.history = History()
        will_validate = bool(validation_data)
        # self.build_model_functions(will_validate)
        build_callbacks = [self.base_logger, self.progbar_logger]
        raw_callbacks = build_callbacks + self.get_callbacks_for_multi_input(temp_best_path, save_weights_only)
        callbacks = CallbackList(raw_callbacks)
        callbacks.set_model(self.model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1,
            'do_validation': will_validate,
            'metrics': self.callbacks_metric,
        })
        return callbacks, will_validate

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
                      save_weights_only: bool = False,
                      will_use_multi_inputs_per_one_image: bool = False,
                      input_data_preprocess_for_building_multi_data=None,
                      output_data_preprocess_for_building_multi_data=None):
        """
        モデルの適合度を算出する
        :param image_generator: ファイルパスから学習データを生成する生成器
        :param epochs: エポック数
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプル
        :param steps_per_epoch:
        :param validation_steps:
        :param temp_best_path:
        :param save_weights_only:
        :param will_use_multi_inputs_per_one_image:
        :param input_data_preprocess_for_building_multi_data:
        :param output_data_preprocess_for_building_multi_data:
        :return:
        """
        print("fit generator")
        self.__model = self.run_preprocess_model(self.__model)
        if validation_data is None:
            if will_use_multi_inputs_per_one_image:
                self.fit_generator_for_multi_inputs_per_one_image(image_generator,
                                                                  epochs=epochs,
                                                                  steps_per_epoch=steps_per_epoch,
                                                                  temp_best_path=temp_best_path,
                                                                  save_weights_only=save_weights_only,
                                                                  input_data_preprocess_for_building_multi_data=input_data_preprocess_for_building_multi_data,
                                                                  output_data_preprocess_for_building_multi_data=output_data_preprocess_for_building_multi_data)
                return self
            callbacks = self.get_callbacks(temp_best_path, save_weights_only)
            if is_new_keras():
                self.__history = self.__model.fit(image_generator,
                                                  steps_per_epoch=steps_per_epoch,
                                                  epochs=epochs,
                                                  callbacks=callbacks)
            else:
                self.__history = self.__model.fit_generator(image_generator,
                                                            steps_per_epoch=steps_per_epoch,
                                                            epochs=epochs,
                                                            callbacks=callbacks)

        else:
            if will_use_multi_inputs_per_one_image:
                self.fit_generator_for_multi_inputs_per_one_image(image_generator,
                                                                  steps_per_epoch=steps_per_epoch,
                                                                  validation_steps=validation_steps,
                                                                  epochs=epochs,
                                                                  validation_data=validation_data,
                                                                  temp_best_path=temp_best_path,
                                                                  save_weights_only=save_weights_only,
                                                                  input_data_preprocess_for_building_multi_data=input_data_preprocess_for_building_multi_data,
                                                                  output_data_preprocess_for_building_multi_data=output_data_preprocess_for_building_multi_data)
                return self
            print('epochs', epochs)
            callbacks = self.get_callbacks(temp_best_path, save_weights_only)
            if is_new_keras():
                self.__history = self.__model.fit(image_generator,
                                                  steps_per_epoch=steps_per_epoch,
                                                  validation_steps=validation_steps,
                                                  epochs=epochs,
                                                  validation_data=validation_data,
                                                  callbacks=callbacks)
            else:
                self.__history = self.__model.fit_generator(image_generator,
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
             save_weights_only: bool = False,
             will_use_multi_inputs_per_one_image: bool = False,
             input_data_preprocess_for_building_multi_data=None,
             output_data_preprocess_for_building_multi_data=None
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
        :param will_use_multi_inputs_per_one_image:
        :param input_data_preprocess_for_building_multi_data:
        :param output_data_preprocess_for_building_multi_data:
        :return:
        """
        write_dir_path = build_record_path(result_dir_name, dir_path)
        save_tmp_name = model_name + "_best.h5" if self.will_save_h5 else model_name + "_best"
        self.fit_generator(image_generator,
                           epochs,
                           validation_data,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           temp_best_path=os.path.join(write_dir_path, save_tmp_name),
                           save_weights_only=save_weights_only,
                           will_use_multi_inputs_per_one_image=will_use_multi_inputs_per_one_image,
                           input_data_preprocess_for_building_multi_data=input_data_preprocess_for_building_multi_data,
                           output_data_preprocess_for_building_multi_data=output_data_preprocess_for_building_multi_data)
        self.record_model(result_dir_name, dir_path, model_name)
        self.record_conf_json(result_dir_name, dir_path, normalize_type, model_name)

    def build_model_functions(self, will_validate):
        self.__model._make_train_function()
        if will_validate:
            self.__model._make_test_function()

    def build_one_batch_dataset(self,
                                output_generator,
                                input_data_preprocess_for_building_multi_data=None,
                                output_data_preprocess_for_building_multi_data=None):
        generator_output = next(output_generator)

        if not hasattr(generator_output, '__len__'):
            raise ValueError('Output of generator should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' +
                             str(generator_output))

        if len(generator_output) == 2:
            x, y = generator_output
            sample_weight = None
        elif len(generator_output) == 3:
            x, y, sample_weight = generator_output
        else:
            raise ValueError('Output of generator should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' +
                             str(generator_output))
        if input_data_preprocess_for_building_multi_data is not None:
            x = input_data_preprocess_for_building_multi_data(x, y)
        if output_data_preprocess_for_building_multi_data is not None:
            y = output_data_preprocess_for_building_multi_data(y)
        return x, y, sample_weight

    def one_batch(self,
                  output_generator,
                  batch_index: int,
                  steps_done: int,
                  callbacks: CallbackList,
                  input_data_preprocess_for_building_multi_data=None,
                  output_data_preprocess_for_building_multi_data=None):
        x, y, sample_weight = self.build_one_batch_dataset(output_generator,
                                                           input_data_preprocess_for_building_multi_data,
                                                           output_data_preprocess_for_building_multi_data)
        # build batch logs
        batch_logs = {}
        callbacks.on_batch_begin(batch_index, batch_logs)

        outs = self.__model.train_on_batch(x,
                                           y,
                                           sample_weight=sample_weight)
        outs = to_list(outs)
        batch_logs["loss"] = outs[0]
        batch_logs["accuracy"] = outs[1]

        callbacks.on_batch_end(batch_index, batch_logs)
        return batch_index+1, steps_done+1

    def one_batch_val(self,
                      val_enqueuer_gen,
                      validation_steps,
                      epoch_logs,
                      input_data_preprocess_for_building_multi_data=None,
                      output_data_preprocess_for_building_multi_data=None):
        steps = len(val_enqueuer_gen)
        steps_done = 0
        outs_per_batch = []
        batch_sizes = []
        print("epoc_logs:", epoch_logs)
        while steps_done < steps:
            x, y, sample_weight = self.build_one_batch_dataset(val_enqueuer_gen,
                                                               input_data_preprocess_for_building_multi_data,
                                                               output_data_preprocess_for_building_multi_data)
            val_outs = self.model.evaluate(x, y, sample_weight=sample_weight)
            val_outs = to_list(val_outs)
            outs_per_batch.append(val_outs)
            if x is None or len(x) == 0:
                # Handle data tensors support when no input given
                # step-size = 1 for data tensors
                batch_size = 1
            elif isinstance(x, list):
                batch_size = x[0].shape[0]
            elif isinstance(x, dict):
                batch_size = list(x.values())[0].shape[0]
            else:
                batch_size = x.shape[0]
            if batch_size == 0:
                raise ValueError('Received an empty batch. '
                                 'Batches should contain '
                                 'at least one item.')
            steps_done += 1
            batch_sizes.append(batch_size)
        losses = [out[0] for out in outs_per_batch]
        accuracies = [out[1] for out in outs_per_batch]
        # Same labels assumed.
        epoch_logs['val_loss'] = np.average(losses, weights=batch_sizes)
        epoch_logs['val_accuracy'] = np.average(accuracies, weights=batch_sizes)
        return epoch_logs

    def build_val_enqueuer(self, validation_data):
        will_validate = bool(validation_data)
        if will_validate is False:
            return None, None, None
        val_data = validation_data
        val_enqueuer = OrderedEnqueuer(
                    val_data,
                    use_multiprocessing=False)
        validation_steps = len(val_data)
        return val_data, val_enqueuer, validation_steps if validation_steps is not None else len(validation_data)

    def run_one_epoch(self,
                      epoch: int,
                      epoch_logs,
                      steps_per_epoch: int,
                      output_generator,
                      val_data,
                      validation_steps,
                      callbacks: CallbackList,
                      input_data_preprocess_for_building_multi_data=None,
                      output_data_preprocess_for_building_multi_data=None):
        # for m in self.model.stateful_metric_functions:
        #   m.reset_states()
        callbacks.on_epoch_begin(epoch)
        steps_done = 0
        batch_index = 0
        while steps_done < steps_per_epoch:
            batch_index, steps_done = self.one_batch(output_generator,
                                                     batch_index,
                                                     steps_done,
                                                     callbacks,
                                                     input_data_preprocess_for_building_multi_data,
                                                     output_data_preprocess_for_building_multi_data)

            # Epoch finished.
            if steps_done >= steps_per_epoch and val_data is not None:
                epoch_logs = self.one_batch_val(val_data,
                                                validation_steps,
                                                epoch_logs,
                                                input_data_preprocess_for_building_multi_data,
                                                output_data_preprocess_for_building_multi_data)

        callbacks.on_epoch_end(epoch, epoch_logs)
        return epoch+1, epoch_logs

    def fit_generator_for_multi_inputs_per_one_image(self,
                                                     image_generator: Union[DataLoaderFromPathsWithDataAugmentation, DataLoaderFromPaths],
                                                     epochs: int,
                                                     validation_data: Union[Optional[Tuple[np.ndarray, np.ndarray]],
                                                                            DataLoaderFromPathsWithDataAugmentation,
                                                                            DataLoaderFromPaths] = None,
                                                     steps_per_epoch: Optional[int] = None,
                                                     validation_steps: Optional[int] = None,
                                                     temp_best_path: str = "",
                                                     save_weights_only: bool = False,
                                                     input_data_preprocess_for_building_multi_data=None,
                                                     output_data_preprocess_for_building_multi_data=None):
        steps_per_epoch = steps_per_epoch if steps_per_epoch is None else len(image_generator)
        callbacks, will_validate = self.build_callbacks_for_multi_input(epochs,
                                                                        temp_best_path,
                                                                        steps_per_epoch,
                                                                        validation_data,
                                                                        save_weights_only)
        callbacks.on_train_begin()
        enqueuer = None
        val_enqueuer = None
        try:
            val_data, val_enqueuer, validation_steps = self.build_val_enqueuer(validation_data)
            enqueuer = OrderedEnqueuer(
                    image_generator,
                    use_multiprocessing=False)
            enqueuer.start(workers=1, max_queue_size=10)
            output_generator = enqueuer.get()

            self.__model.stop_training = False
            # Construct epoch logs.
            epoch_logs = {}
            epoch = 0
            while epoch < epochs:
                epoch, epoch_logs = self.run_one_epoch(epoch,
                                                       epoch_logs,
                                                       steps_per_epoch,
                                                       output_generator,
                                                       val_data,
                                                       validation_steps,
                                                       callbacks,
                                                       input_data_preprocess_for_building_multi_data,
                                                       output_data_preprocess_for_building_multi_data)

        finally:

            if enqueuer is not None:
                print(type(enqueuer))
                enqueuer.stop()
            #finally:
            #    if val_enqueuer is not None:
            #        print(type(val_enqueuer))
            #        val_enqueuer.stop()

        callbacks.on_train_end()
        return self.model.history


