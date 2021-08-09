from keras.callbacks import Callback
from model_merger.pytorch.siamese import SiameseNetworkPT
import numpy as np
import torch


class PytorchCheckpoint(Callback):

    def __init__(self,
                 model: torch.nn.Module,
                 filepath,
                 sample_data,
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto'):
        self.__base_model = model
        self.monitor = monitor
        self.__filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.__sample_data = sample_data

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                self.best = current
                self.__base_model.to("cpu")
                if self.save_weights_only:
                    model_trace = torch.jit.trace(self.__base_model, self.__sample_data)
                    model_trace.save(self.filepath)
                else:
                    model_trace = torch.jit.trace(self.__base_model, self.__sample_data)
                    model_trace.save(self.filepath)
        else:
            self.__base_model.to("cpu")
            if self.save_weights_only:
                model_trace = torch.jit.trace(self.__base_model, self.__sample_data)
                model_trace.save(self.filepath)
            else:
                model_trace = torch.jit.trace(self.__base_model, self.__sample_data)
                model_trace.save(self.filepath)

    @property
    def base_model(self):
        return self.__base_model

    @property
    def filepath(self):
        return self.__filepath


class PytorchSiameseCheckpoint(PytorchCheckpoint):

    def __init__(self,
                 model: SiameseNetworkPT,
                 filepath,
                 sample_data,
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto'):
        super(PytorchSiameseCheckpoint, self).__init__(model,
                                                       filepath,
                                                       sample_data,
                                                       monitor,
                                                       save_best_only,
                                                       save_weights_only,
                                                       mode)

    @property
    def original_model_path(self):
        base_params = self.filepath.split(".")
        return base_params[0] + "_original." + base_params[1]

    def on_epoch_end(self, epoch, logs=None):
        super(PytorchSiameseCheckpoint, self).on_epoch_end(epoch, logs)
        record_model = self.base_model.original_model
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                self.best = current
                record_model.to("cpu")
                if self.save_weights_only:
                    model_trace = torch.jit.trace(record_model, self.__sample_data[0])
                    model_trace.save(self.original_model_path)
                else:
                    model_trace = torch.jit.trace(record_model, self.__sample_data[0])
                    model_trace.save(self.original_model_path)
        else:
            self.__base_model.to("cpu")
            if self.save_weights_only:
                model_trace = torch.jit.trace(self.__base_model, self.__sample_data[0])
                model_trace.save(self.original_model_path)
            else:
                model_trace = torch.jit.trace(self.__base_model, self.__sample_data[0])
                model_trace.save(self.original_model_path)

