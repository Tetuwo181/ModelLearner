from network_model.abstract_model import AbstractModel
from network_model.abstract_expantion_epoch import AbsExpantionEpoch
import keras.callbacks
from typing import List
from typing import Tuple
from typing import Optional
from typing import Union
from typing import Callable
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch import max as torch_max


class ModelForPytorch(AbstractModel, AbsExpantionEpoch):

    def __init__(self,
                 model_base,
                 optimizer: Optimizer,
                 loss: _Loss,
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 will_save_h5: bool = True,
                 preprocess_for_model = None,
                 after_learned_process: Optional[Callable[[None], None]] = None):
        """

        :param model_base: Pytorchで構築したモデル
        :param class_set: クラスの元となったリスト
        :param callbacks: モデルに渡すコールバック関数
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param will_save_h5: 途中モデル読み込み時に旧式のh5ファイルで保存するかどうか　デフォルトだと保存する
        :param preprocess_for_model: モデル学習前にモデルに対してする処理
        :param after_learned_process: モデル学習後の後始末
        """
        self.__model = model_base
        self.__optimizer = optimizer
        self.__loss = loss
        shape = model_base.input[0].shape.as_list() if type(model_base.input) is list else model_base.input.shape.as_list()
        super(ModelForPytorch, self).__init__(shape,
                                              class_set,
                                              callbacks,
                                              monitor,
                                              will_save_h5,
                                              preprocess_for_model,
                                              after_learned_process)

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
        return running_loss, n_correct/n_total

    def add_output_param_to_batch_log_param(self, outs, batch_logs):
        batch_logs["loss"] = outs[0]
        batch_logs["acc"] = outs[1]
        return batch_logs

    def add_output_val_param_to_epoch_log_param(self, outs_per_batch, batch_sizes, epoch_logs):
        pass

    def set_model_stop_training(self, will_stop_trainable):
        pass

    def get_model_history(self):
        pass

    def set_model_history(self):
        pass

    @property
    def callbacks_metric(self):
        pass

    @property
    def model(self) -> keras.engine.training.Model:
        return self.__model