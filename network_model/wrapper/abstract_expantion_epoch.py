from keras.callbacks import CallbackList, ProgbarLogger, BaseLogger, History
from keras.utils.data_utils import Sequence, OrderedEnqueuer
from keras.utils.generic_utils import to_list, unpack_singleton
from abc import ABC, abstractmethod
from network_model.generator import DataLoaderFromPaths
from network_model.generator import DataLoaderFromPathsWithDataAugmentation
from typing import List
from typing import Tuple
from typing import Optional
from typing import Union
from typing import Callable
import numpy as np


class AbsExpantionEpoch(ABC):

    @property
    def stateful_metric_names(self):
        return ["loss", "accuracy", "val_loss", "val_accuracy"]

    @property
    def base_logger(self):
        return BaseLogger(stateful_metrics=self.stateful_metric_names)

    @property
    def progbar_logger(self):
        return ProgbarLogger(count_mode='steps', stateful_metrics=self.stateful_metric_names)

    @abstractmethod
    def train_on_batch(self, x, y, sample_weight=None):
        pass

    @abstractmethod
    def evaluate(self, x, y, sample_weight=None):
        pass

    @abstractmethod
    def add_output_param_to_batch_log_param(self, outs, batch_logs):
        pass

    @abstractmethod
    def add_output_val_param_to_epoch_log_param(self, outs_per_batch, batch_sizes, epoch_logs):
        pass

    @abstractmethod
    def set_model_stop_training(self, will_stop_trainable):
        pass

    @abstractmethod
    def get_model_history(self):
        pass

    @abstractmethod
    def set_model_history(self):
        pass

    @abstractmethod
    def get_callbacks(self, temp_best_path, save_weights_only):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def callbacks_metric(self):
        pass

    def build_one_batch_dataset(self,
                                output_generator,
                                data_preprocess=None):
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
        if data_preprocess is not None:
            x, y = data_preprocess(x, y)
        return x, y, sample_weight

    def get_callbacks_for_expantion(self, temp_best_path, save_weights_only=False):
        base_callbacks = self.get_callbacks(temp_best_path, save_weights_only)
        if base_callbacks is None or base_callbacks == []:
            return [self.get_model_history()]
        return base_callbacks + [self.get_model_history()]

    def build_callbacks_for_expantion(self,
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
        self.set_model_history()
        will_validate = bool(validation_data)
        # self.build_model_functions(will_validate)
        build_callbacks = [self.base_logger, self.progbar_logger]
        raw_callbacks = build_callbacks + self.get_callbacks_for_expantion(temp_best_path, save_weights_only)
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

    def one_batch(self,
                  output_generator,
                  batch_index: int,
                  steps_done: int,
                  callbacks: CallbackList,
                  data_preprocess=None):
        x, y, sample_weight = self.build_one_batch_dataset(output_generator,
                                                           data_preprocess)
        # build batch logs
        batch_logs = {}
        callbacks.on_batch_begin(batch_index, batch_logs)

        outs = self.train_on_batch(x,
                                   y,
                                   sample_weight=sample_weight)
        batch_logs = self.add_output_param_to_batch_log_param(outs, batch_logs)

        callbacks.on_batch_end(batch_index, batch_logs)
        return batch_index+1, steps_done+1

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
                      data_preprocess=None):
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
                                                     data_preprocess)

            # Epoch finished.
            if steps_done >= steps_per_epoch and val_data is not None:
                epoch_logs = self.one_batch_val(val_data,
                                                validation_steps,
                                                epoch_logs,
                                                data_preprocess)

        callbacks.on_epoch_end(epoch, epoch_logs)
        return epoch+1, epoch_logs

    def one_batch_val(self,
                      val_enqueuer_gen,
                      validation_steps,
                      epoch_logs,
                      data_preprocess=None):
        steps = len(val_enqueuer_gen)
        steps_done = 0
        outs_per_batch = []
        batch_sizes = []
        while steps_done < steps:
            try:
                x, y, sample_weight = self.build_one_batch_dataset(val_enqueuer_gen,
                                                                   data_preprocess)
                val_outs = self.evaluate(x, y, sample_weight=sample_weight)
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
            except Exception as e:
                steps_done += 1
                continue
            steps_done += 1
            batch_sizes.append(batch_size)
        return self.add_output_val_param_to_epoch_log_param(outs_per_batch, batch_sizes, epoch_logs)

    def fit_generator_for_expantion(self,
                                    image_generator: Union[DataLoaderFromPathsWithDataAugmentation, DataLoaderFromPaths],
                                    epochs: int,
                                    validation_data: Union[Optional[Tuple[np.ndarray, np.ndarray]],
                                                                          DataLoaderFromPathsWithDataAugmentation,
                                                                          DataLoaderFromPaths] = None,
                                    steps_per_epoch: Optional[int] = None,
                                    validation_steps: Optional[int] = None,
                                    temp_best_path: str = "",
                                    save_weights_only: bool = False,
                                    data_preprocess=None):
        steps_per_epoch = steps_per_epoch if steps_per_epoch is None else len(image_generator)
        callbacks, will_validate = self.build_callbacks_for_expantion(epochs,
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

            self.set_model_stop_training(False)
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
                                                       data_preprocess)

        finally:

            if enqueuer is not None:
                enqueuer.stop()
            #finally:
            #    if val_enqueuer is not None:
            #        print(type(val_enqueuer))
            #        val_enqueuer.stop()

        callbacks.on_train_end()
        return self.get_model_history()
