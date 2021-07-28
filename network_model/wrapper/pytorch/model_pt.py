from network_model.wrapper.abstract_model import AbstractModel
from network_model.wrapper.abstract_expantion_epoch import AbsExpantionEpoch
from keras.callbacks import History
import keras.callbacks
from typing import List
from typing import Optional
import numpy as np
from typing import Callable
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch import max as torch_max
import torch


class ModelForPytorch(AbstractModel, AbsExpantionEpoch):

    @staticmethod
    def build(model_base,
              optimizer: Optimizer,
              loss: _Loss,
              class_set: List[str],
              callbacks: Optional[List[keras.callbacks.Callback]] = None,
              monitor: str = "",
              preprocess_for_model=None,
              after_learned_process: Optional[Callable[[None], None]] = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return ModelForPytorch(model_base,
                               optimizer,
                               loss,
                               device,
                               class_set,
                               callbacks,
                               monitor,
                               preprocess_for_model,
                               after_learned_process
                               )

    def __init__(self,
                 model_base,
                 optimizer: Optimizer,
                 loss: _Loss,
                 torch_device,
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 preprocess_for_model=None,
                 after_learned_process: Optional[Callable[[None], None]] = None):
        """

        :param model_base: Pytorchで構築したモデル
        :param class_set: クラスの元となったリスト
        :param callbacks: モデルに渡すコールバック関数
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param preprocess_for_model: モデル学習前にモデルに対してする処理
        :param after_learned_process: モデル学習後の後始末
        """
        self.__model = model_base
        self.__optimizer = optimizer
        self.__loss = loss
        self.__torch_device = torch_device
        self.__model.to(self.__torch_device)
        shape = model_base.input[0].shape.as_list() if type(model_base.input) is list else model_base.input.shape.as_list()
        super(ModelForPytorch, self).__init__(shape,
                                              class_set,
                                              callbacks,
                                              monitor,
                                              preprocess_for_model,
                                              after_learned_process)

    def numpy2tensor(self, param: np.ndarray) -> torch.tensor:
        converted = torch.from_numpy(param)
        return converted.to(self.__torch_device)

    def train_on_batch(self, x, y, sample_weight=None):
        self.__model.train()
        self.__optimizer.zero_grad()
        outputs = self.__model(x)
        loss = self.__loss(outputs, y)
        loss.backward()
        self.__optimizer.step()
        running_loss = loss.item()
        _, predicted = torch_max(outputs.data, 1)
        n_total = y.size(0)
        n_correct = (predicted == y).sum().item()
        return running_loss, n_correct/n_total

    def evaluate(self, x, y, sample_weight=None):
        self.__model.eval()
        outputs = self.__model(x)
        loss = self.__loss(outputs, y)
        running_loss = loss.item()
        _, predicted = torch_max(outputs.data, 1)
        n_total = y.size(0)
        n_correct = (predicted == y).sum().item()
        return running_loss, n_correct/n_total, n_total

    def add_output_param_to_batch_log_param(self, outs, batch_logs):
        batch_logs["loss"] = outs[0]
        batch_logs["acc"] = outs[1]
        return batch_logs

    def add_output_val_param_to_epoch_log_param(self, outs_per_batch, batch_sizes, epoch_logs):
        losses = [out[0] for out in outs_per_batch]
        epoch_logs['val_loss'] = np.average(losses, weights=batch_sizes)
        if len(outs_per_batch[0]) > 1:
            accuracies = [out[1] for out in outs_per_batch]
            # Same labels assumed.
            epoch_logs['val_accuracy'] = np.average(accuracies, weights=batch_sizes)
        return epoch_logs

    def set_model_stop_training(self, will_stop_trainable):
        pass

    def get_model_history(self):
        return self.__model.history

    def set_model_history(self):
        self.__model.history = History()

    @property
    def callbacks_metric(self):
        return ["loss", "accuracy", "val_loss", "val_accuracy"]

    @property
    def model(self) -> keras.engine.training.Model:
        return self.__model
