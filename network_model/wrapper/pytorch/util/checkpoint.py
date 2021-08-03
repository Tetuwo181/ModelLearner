from keras.callbacks import Callback
import numpy as np
import torch


class PytorchCheckpoint(Callback):

    def __init__(self,
                 model: torch.nn.Module,
                 filepath,
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto'):
        self.__base_model = model
        self.monitor = monitor
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

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
                    torch.save(self.__base_model, self.filepath)
                else:
                    torch.save(self.__base_model, self.filepath)
        else:
            self.__base_model.to("cpu")
            if self.save_weights_only:
                torch.save(self.__base_model, self.filepath)
            else:
                torch.save(self.__base_model, self.filepath)



