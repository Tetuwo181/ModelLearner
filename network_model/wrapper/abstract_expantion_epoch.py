from keras.callbacks import CallbackList, ProgbarLogger, History
from abc import ABC, abstractmethod
from typing import Tuple
from typing import Optional
from typing import Union
import numpy as np


class AbsExpantionEpoch(ABC):

    @property
    def stateful_metric_names(self):
        return ["loss", "accuracy", "val_loss", "val_accuracy"]

    @property
    def progbar_logger(self):
        return ProgbarLogger(count_mode='steps', stateful_metrics=self.stateful_metric_names)

    @abstractmethod
    def train_on_batch(self, x, y, sample_weight=None, data_preprocess=None):
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

    def build_raw_one_batch_dataset(self, output_generator):
        generator_output = next(output_generator)

        if not hasattr(generator_output, '__len__'):
            raise ValueError('Output of builder should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' +
                             str(generator_output))

        if len(generator_output) == 2:
            x, y = generator_output
            sample_weight = None
        elif len(generator_output) == 3:
            x, y, sample_weight = generator_output
        else:
            raise ValueError('Output of builder should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' +
                             str(generator_output))
        return x, y, sample_weight

    def build_one_batch_dataset(self,
                                output_generator,
                                data_preprocess=None,
                                will_get_original: bool = True):
        original_x, original_y, sample_weight = self.build_raw_one_batch_dataset(output_generator)
        if data_preprocess is None:
            return original_x, original_y, sample_weight
        x, y = data_preprocess(original_x, original_y)
        if will_get_original:
            return x, y, sample_weight, original_x, original_y
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
                                      validation_data=None,
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
        build_callbacks = [self.progbar_logger]
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

    def init_for_one_batch(self,
                           output_generator,
                           batch_index: int,
                           callbacks: CallbackList,
                           data_preprocess=None):
        got_params = self.build_one_batch_dataset(output_generator,
                                                  data_preprocess,
                                                  True)
        # build batch logs
        batch_logs = {}
        callbacks.on_batch_begin(batch_index, batch_logs)
        x, y, sample_weight, original_x, original_y = got_params
        return x, y, sample_weight, original_x, original_y, batch_logs

    def run_one_batch_base(self,
                           output_generator,
                           batch_index: int,
                           callbacks: CallbackList,
                           data_preprocess=None):
        x, y, sample_weight, original_x, original_y, batch_logs = self.init_for_one_batch(output_generator,
                                                                                          batch_index,
                                                                                          callbacks,
                                                                                          data_preprocess)
        # build batch logs
        batch_logs = {}
        callbacks.on_batch_begin(batch_index, batch_logs)

        outs = self.train_on_batch(x,
                                   y,
                                   sample_weight=None,
                                   data_preprocess=data_preprocess)
        return x, y,  original_x, original_y, outs, batch_logs

    def run_after_finished_batch(self,
                                 outs,
                                 batch_logs,
                                 callbacks,
                                 batch_index: int,
                                 steps_done: int,
                                 ):
        batch_logs = self.add_output_param_to_batch_log_param(outs, batch_logs)

        callbacks.on_batch_end(batch_index, batch_logs)
        return batch_index+1, steps_done+1

    def one_batch(self,
                  output_generator,
                  batch_index: int,
                  steps_done: int,
                  callbacks: CallbackList,
                  data_preprocess=None):
        x, y, original_x, original_y, outs, batch_logs = self.run_one_batch_base(output_generator,
                                                                                 batch_index,
                                                                                 callbacks,
                                                                                 data_preprocess)
        return self.run_after_finished_batch(outs, batch_logs, callbacks, batch_index, steps_done)

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
                val_outs = np.array(val_outs)
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
                print(e)
                steps_done += 1
                continue
            steps_done += 1
            batch_sizes.append(batch_size)
        return self.add_output_val_param_to_epoch_log_param(outs_per_batch, batch_sizes, epoch_logs)
