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
import torch.nn
from network_model.wrapper.abstract_model import build_record_path
from DataIO import data_loader as dl
import os
import torch.utils.data as data
from torch.nn import BCELoss, CrossEntropyLoss
from network_model.wrapper.pytorch.util.checkpoint import PytorchCheckpoint


class ModelForPytorch(AbstractModel, AbsExpantionEpoch):

    @staticmethod
    def build(model_base: torch.nn.Module,
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

    @staticmethod
    def build_wrapper(model_base: torch.nn.Module,
                      optimizer: Optimizer,
                      loss: _Loss = None):

        def build_model(class_set: List[str],
                        callbacks: Optional[List[keras.callbacks.Callback]] = None,
                        monitor: str = "",
                        preprocess_for_model=None,
                        after_learned_process: Optional[Callable[[None], None]] = None):
            if loss is not None:
                return ModelForPytorch.build(model_base,
                                             optimizer,
                                             loss,
                                             class_set,
                                             callbacks,
                                             monitor,
                                             preprocess_for_model,
                                             after_learned_process)
            use_loss = CrossEntropyLoss() if len(class_set) > 1 else BCELoss()
            return ModelForPytorch.build(model_base,
                                         optimizer,
                                         use_loss,
                                         class_set,
                                         callbacks,
                                         monitor,
                                         preprocess_for_model,
                                         after_learned_process)
        return build_model

    def __init__(self,
                 model_base: torch.nn.Module,
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
        super(ModelForPytorch, self).__init__(class_set,
                                              callbacks,
                                              monitor,
                                              preprocess_for_model,
                                              after_learned_process)

    def numpy2tensor(self, param: np.ndarray, dtype) -> torch.tensor:
        converted = torch.from_numpy(param)
        return converted.to(self.__torch_device, dtype=dtype)

    def convert_data_for_model(self, x: np.ndarray, y):
        data.dataloader.Dataset()
        return self.numpy2tensor(x, torch.float), self.numpy2tensor(y, torch.long)

    def train_on_batch(self, x, y, sample_weight=None):
        self.__model.to(self.__torch_device)
        self.__model.train()
        self.__optimizer.zero_grad()
        x, y = self.convert_data_for_model(x, y)
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
        x, y = self.convert_data_for_model(x, y)
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
        losses = np.array([out[0][0] for out in outs_per_batch])
        epoch_logs['val_loss'] = np.average(losses, weights=batch_sizes)
        if len(outs_per_batch[0][0]) > 1:
            accuracies = [out[0][1] for out in outs_per_batch]
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

    def build_model_file_name(self, model_name):
        return model_name + ".pt"

    def build_best_model_file_name(self, model_name):
        return model_name + "_best.pt"

    def build_write_set(self):
        return {"class_set": self.class_set}

    def test(self,
             image_generator,
             epochs: int,
             validation_data=None,
             normalize_type: dl.NormalizeType = dl.NormalizeType.Div255,
             result_dir_name: str = None,
             dir_path: str = None,
             model_name: str = None,
             steps_per_epoch: Optional[int] = None,
             validation_steps: Optional[int] = None,
             save_weights_only: bool = False,
             will_use_multi_inputs_per_one_image: bool = False,
             input_data_preprocess_for_building_multi_data=None
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
        :return:
        """
        write_dir_path = build_record_path(result_dir_name, dir_path)
        save_tmp_name = self.build_best_model_file_name(model_name)
        self.fit_generator(image_generator,
                           epochs,
                           validation_data,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           temp_best_path=os.path.join(write_dir_path, save_tmp_name),
                           save_weights_only=save_weights_only,
                           will_use_multi_inputs_per_one_image=will_use_multi_inputs_per_one_image,
                           data_preprocess=input_data_preprocess_for_building_multi_data)
        self.record_model(result_dir_name, dir_path, model_name)
        self.record_conf_json(result_dir_name, dir_path, normalize_type, model_name)

    def fit_generator(self,
                      image_generator,
                      epochs: int,
                      validation_data=None,
                      steps_per_epoch: Optional[int] = None,
                      validation_steps: Optional[int] = None,
                      temp_best_path: str = "",
                      save_weights_only: bool = False,
                      will_use_multi_inputs_per_one_image: bool = False,
                      data_preprocess=None):
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
        :param data_preprocess:
        :return:
        """
        print("fit builder")
        self.__model = self.run_preprocess_model(self.__model)
        self.__model.to(self.__torch_device)
        if validation_data is None:
            self.fit_generator_for_expantion(image_generator,
                                             epochs=epochs,
                                             steps_per_epoch=steps_per_epoch,
                                             temp_best_path=temp_best_path,
                                             save_weights_only=save_weights_only,
                                             data_preprocess=data_preprocess)
        else:
            self.fit_generator_for_expantion(image_generator,
                                             steps_per_epoch=steps_per_epoch,
                                             validation_steps=validation_steps,
                                             epochs=epochs,
                                             validation_data=validation_data,
                                             temp_best_path=temp_best_path,
                                             save_weights_only=save_weights_only,
                                             data_preprocess=data_preprocess)

        return self

    def save_model(self, file_path):
        self.__model.to("cpu")
        torch.save(self.__model, file_path)

    def build_model_checkpoint(self, temp_best_path, save_weights_only):
        return PytorchCheckpoint(self.__model,
                                 temp_best_path,
                                 monitor=self.monitor,
                                 save_best_only=True,
                                 save_weights_only=save_weights_only)



