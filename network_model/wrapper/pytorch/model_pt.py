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
from torch.nn import BCELoss, CrossEntropyLoss
from network_model.wrapper.pytorch.util.checkpoint import PytorchCheckpoint, PytorchSiameseCheckpoint
from model_merger.pytorch.siamese import SiameseNetworkPT
from keras.utils.generic_utils import to_list
from generator.transpose import transpose
from generator.siamese_learner import SiameseLearnerDataBuilder
from generator.siamese_learner_for_inceptionv3_age import SiameseLearnerDataBuilderForInceptionV3
from model_merger.pytorch.proc.shiamese_loss import SiameseLossForInceptionV3
from torchvision.models.inception import Inception3
from network_model.wrapper.pytorch.util.neighbor_recorder import NeighborRecorder
from numba import jit


class ModelForPytorch(AbstractModel, AbsExpantionEpoch):

    @staticmethod
    def build_sampledata(is_siamese: bool):
        base_data = torch.rand(1, 3, 224, 224)
        return [base_data, base_data] if is_siamese else base_data

    @staticmethod
    def build(model_base: torch.nn.Module,
              optimizer: Optimizer,
              loss: _Loss,
              class_set: List[str],
              callbacks: Optional[List[keras.callbacks.Callback]] = None,
              monitor: str = "",
              preprocess_for_model=None,
              after_learned_process: Optional[Callable[[None], None]] = None,
              sample_data=None,
              x_type=torch.float,
              y_type=None,
              decide_dataset_generator=None,
              nearest_data_ave_num=1,
              will_calc_rate_real_data_train=False):
        use_sample_data = sample_data
        if use_sample_data is None:
            use_sample_data = ModelForPytorch.build_sampledata(isinstance(model_base, SiameseNetworkPT))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if decide_dataset_generator is not None:
            return ModelForPytorchSiameseDecidebyDistance(model_base,
                                                          optimizer,
                                                          loss,
                                                          device,
                                                          class_set,
                                                          decide_dataset_generator,
                                                          callbacks,
                                                          monitor,
                                                          preprocess_for_model,
                                                          after_learned_process,
                                                          use_sample_data,
                                                          x_type,
                                                          y_type,
                                                          nearest_data_ave_num,
                                                          will_calc_rate_real_data_train
                                                          )
        if isinstance(model_base, ModelForPytorchSiameseInceptionV3):
            return ModelForPytorchSiameseInceptionV3(model_base,
                                                     optimizer,
                                                     loss,
                                                     device,
                                                     class_set,
                                                     callbacks,
                                                     monitor,
                                                     preprocess_for_model,
                                                     after_learned_process,
                                                     use_sample_data,
                                                     x_type,
                                                     y_type
                                                     )
        if isinstance(model_base, ModelForPytorchSiamese):
            return ModelForPytorchSiamese(model_base,
                                          optimizer,
                                          loss,
                                          device,
                                          class_set,
                                          callbacks,
                                          monitor,
                                          preprocess_for_model,
                                          after_learned_process,
                                          use_sample_data,
                                          x_type,
                                          y_type
                                          )
        return ModelForPytorch(model_base,
                               optimizer,
                               loss,
                               device,
                               class_set,
                               callbacks,
                               monitor,
                               preprocess_for_model,
                               after_learned_process,
                               use_sample_data,
                               x_type,
                               y_type
                               )

    @staticmethod
    def build_builder(model_base: torch.nn.Module,
                      optimizer: Optimizer,
                      loss: _Loss,
                      class_set: List[str],
                      callbacks: Optional[List[keras.callbacks.Callback]] = None,
                      monitor: str = "",
                      preprocess_for_model=None,
                      after_learned_process: Optional[Callable[[None], None]] = None,
                      sample_data=None,
                      x_type=torch.float,
                      y_type=None,
                      teacher_dataset=None,
                      decide_dataset_generator=None,
                      nearest_data_ave_num=1):
        if teacher_dataset is None:
            return ModelForPytorch.build(model_base,
                                         optimizer,
                                         loss,
                                         class_set,
                                         callbacks,
                                         monitor,
                                         preprocess_for_model,
                                         after_learned_process,
                                         sample_data,
                                         x_type,
                                         y_type,
                                         decide_dataset_generator,
                                         nearest_data_ave_num)



    @staticmethod
    def build_wrapper(model_base: torch.nn.Module,
                      optimizer: Optimizer,
                      loss: _Loss = None,
                      sample_data=None,
                      decide_dataset_generator=None,
                      nearest_data_ave_num=1,
                      will_calc_rate_real_data_train=False):
        use_sample_data = sample_data
        if use_sample_data is None:
            use_sample_data = ModelForPytorch.build_sampledata(isinstance(model_base, SiameseNetworkPT))

        def build_model(class_set: List[str],
                        callbacks: Optional[List[keras.callbacks.Callback]] = None,
                        monitor: str = "",
                        preprocess_for_model=None,
                        after_learned_process: Optional[Callable[[None], None]] = None,
                        x_type=torch.float,
                        y_type=None):
            if loss is not None:
                return ModelForPytorch.build(model_base,
                                             optimizer,
                                             loss,
                                             class_set,
                                             callbacks,
                                             monitor,
                                             preprocess_for_model,
                                             after_learned_process,
                                             use_sample_data,
                                             x_type,
                                             y_type,
                                             decide_dataset_generator=decide_dataset_generator,
                                             nearest_data_ave_num=nearest_data_ave_num,
                                             will_calc_rate_real_data_train=will_calc_rate_real_data_train)
            use_loss = CrossEntropyLoss() if len(class_set) > 2 else BCELoss()
            return ModelForPytorch.build(model_base,
                                         optimizer,
                                         use_loss,
                                         class_set,
                                         callbacks,
                                         monitor,
                                         preprocess_for_model,
                                         after_learned_process,
                                         use_sample_data,
                                         x_type,
                                         y_type,
                                         decide_dataset_generator=decide_dataset_generator,
                                         nearest_data_ave_num=nearest_data_ave_num,
                                         will_calc_rate_real_data_train=will_calc_rate_real_data_train)
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
                 after_learned_process: Optional[Callable[[None], None]] = None,
                 sample_data=torch.rand(1, 3, 224, 224),
                 x_type=torch.float,
                 y_type=None):
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
        self.__x_type = x_type
        self.__y_type = y_type
        self.__sample_data = sample_data
        if self.__y_type is None:
            self.__y_type = torch.long if len(class_set) > 2 else torch.float
        super(ModelForPytorch, self).__init__(class_set,
                                              callbacks,
                                              monitor,
                                              preprocess_for_model,
                                              after_learned_process)

    @property
    def is_binary_classifier(self):
        return self.__y_type==torch.float

    @property
    def is_siamese(self):
        return isinstance(self.__model, SiameseNetworkPT)

    @property
    def is_inceptionV3(self):
        return isinstance(self.__model, Inception3)

    @property
    def is_siamese_inceptionV3(self):
        if self.is_siamese is False:
            return False
        return isinstance(self.__loss, SiameseLossForInceptionV3)

    @property
    def model(self):
        return self.__model

    @property
    def loss(self):
        return self.__loss

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def torch_device(self):
        return self.__torch_device

    @property
    def sample_data(self):
        return self.__sample_data

    @property
    def x_type(self):
        return self.__x_type

    @property
    def y_type(self):
        return self.__y_type

    @property
    def stateful_metric_names(self):
        if self.is_siamese_inceptionV3:
            return ["loss",
                    "accuracy",
                    "aux_loss",
                    "aux_accuracy",
                    "original_accuracy",
                    "aux_original_accuracy",
                    "val_loss",
                    "val_accuracy",
                    "val_original_accuracy"]
        if self.is_siamese:
            return ["loss", "accuracy", "val_loss", "val_accuracy", "val_original_accuracy"]
        return ["loss", "accuracy", "val_loss", "val_accuracy"]

    def set_model_to_device(self):
        return self.__model.to(self.__torch_device)

    def become_train_mode(self):
        return self.__model.train()

    def become_eval_mode(self):
        return self.__model.eval()

    def numpy2tensor(self, param: np.ndarray, dtype) -> torch.tensor:
        converted = torch.from_numpy(param)
        return converted.to(self.__torch_device, dtype=dtype)

    def convert_data_for_model(self, x: np.ndarray, y):
        return self.numpy2tensor(x, self.__x_type), self.numpy2tensor(y, self.__y_type)

    def train_on_batch(self, x, y, sample_weight=None, data_preprocess=None):
        self.__model.to(self.__torch_device)
        self.__model.train()
        self.__optimizer.zero_grad()
        x, y = self.convert_data_for_model(x, y)
        outputs = self.__model(x)
        if self.is_siamese_inceptionV3:
            loss, aux_loss = self.__loss(outputs, y)
            loss.backward(retain_graph=True)
            running_loss = loss.item()
            aux_loss.backward(retain_graph=True)
            aux_running_loss = aux_loss.item()
            self.__optimizer.step()
            predicted, aux_predicted = self.get_predicted(outputs)
            collect_rate = self.calc_collect_rate(predicted, y[0])
            aux_collect_rate = self.calc_collect_rate(aux_predicted, y[1])
            return running_loss, collect_rate, aux_running_loss, aux_collect_rate
        if self.is_inceptionV3:
            loss = self.__loss(outputs.logits, y)
            self.__optimizer.step()
            running_loss = loss.item()
            predicted = self.get_predicted(outputs.logits)
            if outputs.aux_logits is not None:
                aux_loss = self.__loss(outputs.aux_logits, y)
                self.__optimizer.step()
            return running_loss, self.calc_collect_rate(predicted, y)
        loss = self.__loss(outputs, y)
        loss.backward()
        self.__optimizer.step()
        running_loss = loss.item()
        predicted = self.get_predicted(outputs)
        self.__sample_data = x[:1].to("cpu")
        return running_loss, self.calc_collect_rate(predicted, y)

    def get_siamese_predicted(self, x0, x1):
        distance = self.__loss.calc_distance(x0, x1)
        return 0 if distance < 0.5 else 1

    def get_siamese_predicted_batch(self, outputs):
        x0, x1 = outputs
        return [self.get_siamese_predicted(param0, param1) for param0, param1 in zip(x0, x1)]

    def get_predicted(self, outputs, is_training: bool = True):
        if self.is_siamese_inceptionV3:
            if is_training is False:
                return self.get_siamese_predicted_batch(outputs)
            x0, x1 = outputs
            predicted = [self.get_siamese_predicted(param0, param1) for param0, param1 in zip(x0.logits, x1.logits)]
            aux_predicted = [self.get_siamese_predicted(param0, param1) for param0, param1
                             in zip(x0.aux_logits, x1.aux_logits)]
            return predicted, aux_predicted

        if self.is_siamese:
            return self.get_siamese_predicted_batch(outputs)
        if self.is_binary_classifier:
            return [0 if param < 0.5 else 1 for param in outputs.data]
        _, predicted = torch_max(outputs.data, 1)
        return predicted

    def calc_collect_rate(self, predicted, y):
        n_total = y.size(0)
        if self.is_siamese:
            correct_num = 0
            for predicted_param, teacher in zip(predicted, y):
                if teacher < 0.5 and predicted_param < 0.5:
                    correct_num = correct_num + 1
                if teacher > 0.5 and predicted_param > 0.5:
                    correct_num = correct_num + 1
            return correct_num/n_total
        if self.is_binary_classifier:
            correct_num = 0
            for predicted_param, teacher in zip(predicted, y):
                if teacher[0] < 0.5 and predicted_param < 0.5:
                    correct_num = correct_num + 1
                if teacher[0] > 0.5 and predicted_param > 0.5:
                    correct_num = correct_num + 1
            return correct_num/n_total
        correct_num = (predicted == y).sum().item()
        return correct_num/n_total

    def build_evaluate_output(self, x, y):
        self.__model.eval()
        x, y = self.convert_data_for_model(x, y)
        outputs = self.__model(x)
        return outputs, x, y

    def evaluate(self, x, y, sample_weight=None):
        outputs, x, y = self.build_evaluate_output(x, y)
        loss = self.__loss(outputs, y, True)
        running_loss = loss.item()
        predicted = self.get_predicted(outputs, False)
        return running_loss, self.calc_collect_rate(predicted, y)

    def add_output_param_to_batch_log_param(self, outs, batch_logs):
        batch_logs["loss"] = outs[0]
        batch_logs["accuracy"] = outs[1]
        if self.is_siamese_inceptionV3:
            batch_logs["aux_loss"] = outs[2]
            batch_logs["aux_accuracy"] = outs[3]
        return batch_logs

    def add_output_val_param_to_epoch_log_param(self, outs_per_batch, batch_sizes, epoch_logs):
        losses = np.array([out[0][0] for out in outs_per_batch])
        epoch_logs['val_loss'] = np.average(losses, weights=batch_sizes)
        if len(outs_per_batch[0][0]) > 1:
            accuracies = [out[0][1] for out in outs_per_batch]
            # Same labels assumed.
            epoch_logs['val_accuracy'] = np.average(accuracies, weights=batch_sizes)
        if self.is_siamese:
            original_accuracies = [out[0][2] for out in outs_per_batch]
            epoch_logs['val_original_accuracy'] = np.average(original_accuracies, weights=batch_sizes)
        return epoch_logs

    def set_model_stop_training(self, will_stop_trainable):
        pass

    def get_model_history(self):
        return self.__model.history

    def set_model_history(self):
        self.__model.history = History()

    @property
    def callbacks_metric(self):
        if self.is_siamese_inceptionV3:
            return ["loss",
                    "accuracy",
                    "aux_loss",
                    "aux_accuracy",
                    "original_accuracy",
                    "aux_original_accuracy",
                    "val_loss",
                    "val_accuracy",
                    "val_aux_loss",
                    "val_aux_accuracy",
                    "val_original_accuracy"]
        if self.is_siamese:
            return ["loss", "accuracy", "val_loss", "val_accuracy", "val_original_accuracy"]
        return ["loss", "accuracy", "val_loss", "val_accuracy"]


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
        if self.is_siamese:
            return PytorchSiameseCheckpoint(self.model,
                                            temp_best_path,
                                            self.sample_data,
                                            monitor=self.monitor,
                                            save_best_only=True,
                                            save_weights_only=save_weights_only)
        return PytorchCheckpoint(self.model,
                                 temp_best_path,
                                 self.sample_data,
                                 monitor=self.monitor,
                                 save_best_only=True,
                                 save_weights_only=save_weights_only)

    def evaluate_siamese_build_original_output(self,
                                               original_x,
                                               original_y):
        original_model = self.model.original_model
        original_model.eval()
        original_model.to(self.__torch_device)
        converted_x = self.numpy2tensor(transpose(original_x), self.__x_type)
        converted_y = self.numpy2tensor(original_y, torch.long)
        original_output = original_model(converted_x)
        return original_output, converted_y

    def calc_original_rate_for_siamese(self,
                                       predicted_data,
                                       original_y,
                                       margin):
        _, predicted = torch_max(predicted_data, 1)
        abstract_predict = torch.abs(predicted - original_y)
        correct_num = (abstract_predict < margin).sum().item()
        n_total = original_y.size(0)
        return correct_num/n_total

    def evaluate_siamese(self,
                         siamese_x,
                         siamese_y,
                         original_x,
                         original_y,
                         steps_done: int,
                         sample_weight=None,
                         data_preprocess=None):
        margin = data_preprocess.margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
        aux_margin = data_preprocess.aux_margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
        original_output, original_y = self.evaluate_siamese_build_original_output(original_x, original_y)
        correct_rate = self.calc_original_rate_for_siamese(original_output.data, original_y, margin)
        use_siamese_y = siamese_y[0] if self.is_siamese_inceptionV3 else siamese_y
        running_loss, siamese_collect_rate = self.evaluate(siamese_x, use_siamese_y, sample_weight)
        return running_loss, siamese_collect_rate, correct_rate

    def one_batch_val(self,
                      val_enqueuer_gen,
                      validation_steps,
                      epoch_logs,
                      data_preprocess=None):
        if self.is_siamese is False:
            return super(ModelForPytorch, self).one_batch_val(val_enqueuer_gen,
                                                              validation_steps,
                                                              epoch_logs,
                                                              data_preprocess)
        steps = len(val_enqueuer_gen)
        steps_done = 0
        outs_per_batch = []
        batch_sizes = []
        print(type(data_preprocess))
        while steps_done < steps:
            try:
                print("step", steps_done)
                siamese_x, siamese_y, sample_weight, x, y = self.build_one_batch_dataset(val_enqueuer_gen,
                                                                                         data_preprocess,
                                                                                         False)
                if isinstance(data_preprocess, SiameseLearnerDataBuilderForInceptionV3):
                    x, y = data_preprocess.preprocess_evaluate_original(x, y)
                val_outs = self.evaluate_siamese(siamese_x,
                                                 siamese_y,
                                                 x,
                                                 y,
                                                 steps_done,
                                                 sample_weight=sample_weight,
                                                 data_preprocess=data_preprocess)
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
                print(e)
                steps_done += 1
                continue
            steps_done += 1
            batch_sizes.append(batch_size)
        return self.add_output_val_param_to_epoch_log_param(outs_per_batch, batch_sizes, epoch_logs)


class ModelForPytorchSiamese(ModelForPytorch):

    @property
    def callbacks_metric(self):
        return ["loss", "accuracy", "val_loss", "val_accuracy", "val_original_accuracy"]

    def add_output_val_param_to_epoch_log_param(self, outs_per_batch, batch_sizes, epoch_logs):
        losses = np.array([out[0][0] for out in outs_per_batch])
        epoch_logs['val_loss'] = np.average(losses, weights=batch_sizes)
        accuracies = [out[0][1] for out in outs_per_batch]
        # Same labels assumed.
        epoch_logs['val_accuracy'] = np.average(accuracies, weights=batch_sizes)
        original_accuracies = [out[0][2] for out in outs_per_batch]
        epoch_logs['val_original_accuracy'] = np.average(original_accuracies, weights=batch_sizes)
        return epoch_logs

    def build_model_checkpoint(self, temp_best_path, save_weights_only):
        return PytorchSiameseCheckpoint(self.model,
                                        temp_best_path,
                                        self.sample_data,
                                        monitor=self.monitor,
                                        save_best_only=True,
                                        save_weights_only=save_weights_only)

    def build_one_batch_dataset(self,
                                output_generator,
                                data_preprocess=None,
                                will_get_original: bool = True):
        if will_get_original:
            return super(ModelForPytorchSiamese, self).build_one_batch_dataset(output_generator,
                                                                               data_preprocess)
        x, y, sample_weight = self.build_raw_one_batch_dataset(output_generator)
        siamese_x, siamese_y = data_preprocess(x, y)
        return siamese_x, siamese_y, sample_weight, x, y

    def calc_collect_rate(self, predicted, y):
        n_total = y.size(0)
        correct_num = 0
        for predicted_param, teacher in zip(predicted, y):
            if teacher < 0.5 and predicted_param < 0.5:
                correct_num = correct_num + 1
            if teacher > 0.5 and predicted_param > 0.5:
                correct_num = correct_num + 1
        return correct_num/n_total

    def get_predicted(self, outputs, is_training: bool = True):
        if is_training is False:
            return self.get_siamese_predicted_batch(outputs)
        x0, x1 = outputs
        predicted = [self.get_siamese_predicted(param0, param1) for param0, param1 in zip(x0.logits, x1.logits)]
        aux_predicted = [self.get_siamese_predicted(param0, param1) for param0, param1
                         in zip(x0.aux_logits, x1.aux_logits)]
        return predicted, aux_predicted

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
                siamese_x, siamese_y, sample_weight, x, y = self.build_one_batch_dataset(val_enqueuer_gen,
                                                                                         data_preprocess,
                                                                                         False)
                margin = data_preprocess.margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
                aux_margin = data_preprocess.aux_margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
                if isinstance(data_preprocess, SiameseLearnerDataBuilderForInceptionV3):
                    x, y = data_preprocess.preprocess_evaluate_original(x, y)
                val_outs = self.evaluate_siamese(siamese_x,
                                                 siamese_y,
                                                 x,
                                                 y,
                                                 margin,
                                                 steps_done,
                                                 sample_weight=sample_weight,
                                                 aux_margin=aux_margin,
                                                 data_preprocess=data_preprocess)
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
                print(e)
                steps_done += 1
                continue
            steps_done += 1
            batch_sizes.append(batch_size)
        return self.add_output_val_param_to_epoch_log_param(outs_per_batch, batch_sizes, epoch_logs)


class ModelForPytorchSiameseInceptionV3(ModelForPytorch):

    @property
    def callbacks_metric(self):
        return ["loss",
                "accuracy",
                "aux_loss",
                "aux_accuracy",
                "val_loss",
                "val_accuracy",
                "val_aux_loss",
                "val_aux_accuracy",
                "val_original_accuracy"]

    @property
    def stateful_metric_names(self):
        return ["loss",
                "accuracy",
                "aux_loss",
                "aux_accuracy",
                "val_loss",
                "val_accuracy",
                "val_original_accuracy"]

    def add_output_param_to_batch_log_param(self, outs, batch_logs):
        batch_logs["loss"] = outs[0]
        batch_logs["accuracy"] = outs[1]
        batch_logs["aux_loss"] = outs[2]
        batch_logs["aux_accuracy"] = outs[3]
        return batch_logs

    def evaluate_siamese(self,
                         siamese_x,
                         siamese_y,
                         original_x,
                         original_y,
                         steps_done: int,
                         sample_weight=None,
                         data_preprocess=None):
        margin = data_preprocess.margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
        aux_margin = data_preprocess.aux_margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
        original_output, original_y = self.evaluate_siamese_build_original_output(original_x, original_y)
        correct_rate = self.calc_original_rate_for_siamese(original_output.data, original_y, margin)
        use_siamese_y = siamese_y[0]
        running_loss, siamese_collect_rate = self.evaluate(siamese_x, use_siamese_y, sample_weight)
        return running_loss, siamese_collect_rate, correct_rate

    def get_predicted(self, outputs, is_training: bool = True):
        if is_training is False:
            return self.get_siamese_predicted_batch(outputs)
        x0, x1 = outputs
        predicted = [self.get_siamese_predicted(param0, param1) for param0, param1 in zip(x0.logits, x1.logits)]
        aux_predicted = [self.get_siamese_predicted(param0, param1) for param0, param1
                         in zip(x0.aux_logits, x1.aux_logits)]
        return predicted, aux_predicted

    def train_on_batch(self,
                       x,
                       y,
                       sample_weight=None,
                       data_preprocess=None):
        self.set_model_to_device()
        self.become_train_mode()
        self.optimizer.zero_grad()
        x, y = self.convert_data_for_model(x, y)
        outputs = self.model(x)
        loss, aux_loss = self.loss(outputs, y)
        loss.backward(retain_graph=True)
        running_loss = loss.item()
        aux_loss.backward(retain_graph=True)
        aux_running_loss = aux_loss.item()
        self.__optimizer.step()
        predicted, aux_predicted = self.get_predicted(outputs)
        collect_rate = self.calc_collect_rate(predicted, y[0])
        aux_collect_rate = self.calc_collect_rate(aux_predicted, y[1])
        return running_loss, collect_rate, aux_running_loss, aux_collect_rate


class ModelForPytorchSiameseDecidebyDistance(ModelForPytorchSiamese):

    def __init__(self,
                 model_base: torch.nn.Module,
                 optimizer: Optimizer,
                 loss: _Loss,
                 torch_device,
                 class_set: List[str],
                 decide_dataset_generator,
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 preprocess_for_model=None,
                 after_learned_process: Optional[Callable[[None], None]] = None,
                 sample_data=torch.rand(1, 3, 224, 224),
                 x_type=torch.float,
                 y_type=None,
                 nearest_data_ave_num=1,
                 will_calc_rate_real_data_train=False):

        super(ModelForPytorchSiameseDecidebyDistance, self).__init__(model_base,
                                                                     optimizer,
                                                                     loss,
                                                                     torch_device,
                                                                     class_set,
                                                                     callbacks,
                                                                     monitor,
                                                                     preprocess_for_model,
                                                                     after_learned_process,
                                                                     sample_data,
                                                                     x_type,
                                                                     y_type)
        self.__decide_dataset_generator = decide_dataset_generator
        self.__nearest_data_ave_num = nearest_data_ave_num
        self.__will_calc_real_data_train = will_calc_rate_real_data_train
        self.__memorize_predicted_results = None
        self.__memorize_teachers = None

    def evaluate_siamese(self,
                         siamese_x,
                         siamese_y,
                         original_x,
                         original_y,
                         steps_done: int,
                         sample_weight=None,
                         data_preprocess=None):
        self.model.eval()
        correct_rate = self.build_calc_succeed_rate_dataset(siamese_x[0],
                                                            original_y,
                                                            data_preprocess,
                                                            steps_done)
        use_siamese_y = siamese_y[0] if self.is_siamese_inceptionV3 else siamese_y
        running_loss, siamese_collect_rate = self.evaluate(siamese_x, use_siamese_y, sample_weight)
        return running_loss, siamese_collect_rate, correct_rate

    def build_calc_succeed_rate_dataset(self,
                                        x: np.ndarray,
                                        y: np.ndarray,
                                        data_preprocess,
                                        steps_done: int,
                                        is_training=False):
        margin = data_preprocess.margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
        base_predicted = self.model.get_original_predict(self.numpy2tensor(x, self.x_type))
        sample_predicted, sample_teacher = self.get_predict_sample_data(data_preprocess, steps_done, is_training)
        if is_training:
            main_predicted = self.get_classes_from_distances(base_predicted.logits.cpu().detach().numpy().copy(),
                                                             sample_predicted[0],
                                                             sample_teacher[0])
            aux_margin = data_preprocess.aux_margin if isinstance(data_preprocess, SiameseLearnerDataBuilder) else 1
            aux_predicted = self.get_classes_from_distances(base_predicted.aux_logits.cpu().detach().numpy().copy(),
                                                            sample_predicted[1],
                                                            sample_teacher[1])
            teachers = data_preprocess.build_teachers_for_train(y)
            diff = main_predicted - teachers[0]
            main_rate = len(diff[diff < margin])/len(diff)
            aux_diff = aux_predicted - teachers[1]
            aux_rate = len(aux_diff[aux_diff < aux_margin])/len(aux_diff)
            return main_rate, aux_rate
        predicted = self.get_classes_from_distances(base_predicted.cpu().detach().numpy().copy(),
                                                    sample_predicted,
                                                    sample_teacher)
        teacher = data_preprocess.build_teachers_for_train(y)[0]
        diff = np.abs(predicted-teacher)
        correct_num = (diff < margin).sum()
        return correct_num/diff.size

    def get_pair_for_predict_input(self, x, predict_dataset_batch):
        input_x = np.array([x for _ in predict_dataset_batch])
        converted_input = self.numpy2tensor(input_x, self.x_type)
        predict_dataset_batch = self.numpy2tensor(predict_dataset_batch, self.x_type)
        return [converted_input, predict_dataset_batch]

    def decide_class_from_distance(self, distances, decide_batch_y, neighbor_recorder=None):
        use_recorder = NeighborRecorder(self.__nearest_data_ave_num) if neighbor_recorder is None else neighbor_recorder
        for distance, class_index in zip(distances, decide_batch_y):
            use_recorder.record(distance, class_index)
        return use_recorder

    def calc_result_distances_for_train_mode(self, predicted_result):
        out0, out1 = predicted_result
        main_distances = self.loss.calc_distance(out0.logits, out1.logits)
        aux_distances = self.loss.calc_distance(out0.aux_logits, out1.aux_logits)
        return main_distances.cpu().detach().numpy().copy(), aux_distances.cpu().detach().numpy().copy()

    def get_predicted_from_a_data_train_mode(self, x, data_preprocess):
        neigbor_recorder = NeighborRecorder(self.__nearest_data_ave_num)
        aux_neighbor_recorder = NeighborRecorder(self.__nearest_data_ave_num)
        index = 0
        max_index = len(self.__decide_dataset_generator)
        while index < max_index:
            decide_batch_x, decide_batch_y = next(self.__decide_dataset_generator)
            decide_batch_x, decide_batch_y = data_preprocess.preprocess_for_calc_data(decide_batch_x,
                                                                                      decide_batch_y,
                                                                                      True)
            use_batch = self.get_pair_for_predict_input(x, decide_batch_x)
            predicted_result = self.model(use_batch)
            main_distances, aux_distances = self.calc_result_distances_for_train_mode(predicted_result)
            neigbor_recorder = self.decide_class_from_distance(main_distances,
                                                               decide_batch_y[0],
                                                               neigbor_recorder)
            aux_neighbor_recorder = self.decide_class_from_distance(aux_distances,
                                                                    decide_batch_y[1],
                                                                    aux_neighbor_recorder)
            index = index+1
        return neigbor_recorder.get_predicted_index(), aux_neighbor_recorder.get_predicted_index()

    def get_classes_from_distances(self, predict, sample_predicted, sample_teacher):
        return np.array([self.calc_class_from_distance(param, sample_predicted, sample_teacher) for param in predict])

    def calc_class_from_distance(self, base_predict, sample_predicted, sample_teacher):
        use_base_predict_set = torch.from_numpy(np.array([base_predict for _ in sample_predicted]))
        distances = self.loss.calc_distance(use_base_predict_set, sample_predicted).cpu().detach().numpy()
        return self.decide_class_from_distance(distances, sample_teacher).get_predicted_index()

    def get_predicted_from_a_data(self, x, data_preprocess, is_training: bool = False):
        if is_training:
            return self.get_predicted_from_a_data_train_mode(x, data_preprocess)
        neighbor_recorder = NeighborRecorder(self.__nearest_data_ave_num)
        index = 0
        max_index = len(self.__decide_dataset_generator)
        while index < max_index:
            decide_batch_x, decide_batch_y = next(self.__decide_dataset_generator)
            decide_batch_x, decide_batch_y = data_preprocess.preprocess_for_calc_data(decide_batch_x, decide_batch_y)
            use_batch = self.get_pair_for_predict_input(x, decide_batch_x)
            predicted_result = self.model(use_batch)
            distances = self.loss.calc_distance(predicted_result[0], predicted_result[1])
            distances = distances.cpu().detach().numpy().copy()
            for distance, class_index in zip(distances, decide_batch_y):
                neighbor_recorder.record(distance, class_index)
            index = index + 1
        return neighbor_recorder.get_predicted_index()

    def get_predict_sample_data(self, data_preprocess, steps_done, is_training=False):
        if steps_done > 0 and is_training is not True:
            return self.__memorize_predicted_results, self.__memorize_teachers
        index = 0
        max_index = len(self.__decide_dataset_generator)
        while index < max_index:
            decide_batch_x, decide_batch_y = next(self.__decide_dataset_generator)
            decide_batch_x, decide_batch_y = data_preprocess.preprocess_for_calc_data(decide_batch_x,
                                                                                      decide_batch_y,
                                                                                      is_training)
            predicted = self.model.get_original_predict(self.numpy2tensor(decide_batch_x, self.x_type))
            if is_training:
                predicted = (predicted.logits.cpu().detach().numpy(), predicted.aux_logits.cpu().detach().numpy())
                predicted_results = predicted if index == 0 else (np.append(predicted_results[0], predicted[0], axis=0), np.append(predicted_results[1], predicted[1], axis=0))
                teachers = decide_batch_y if index == 0 else (np.append(teachers[0], decide_batch_y[0], axis=0), np.append(teachers[1], decide_batch_y[1], axis=0))
            else:
                predicted = predicted.cpu().detach().numpy()
                predicted_results = predicted if index == 0 else np.append(predicted_results, predicted, axis=0)
                teachers = decide_batch_y if index == 0 else np.append(teachers, decide_batch_y, axis=0)
            index = index+1
            self.__memorize_predicted_results = predicted_results
            self.__memorize_teachers = teachers
        return self.__memorize_predicted_results, self.__memorize_teachers

    def one_batch(self,
                  output_generator,
                  batch_index: int,
                  steps_done: int,
                  callbacks,
                  data_preprocess=None):
        x, y, original_x, original_y, outs, batch_logs = self.run_one_batch_base(output_generator,
                                                                                 batch_index,
                                                                                 callbacks,
                                                                                 data_preprocess)
        if self.__will_calc_real_data_train:
            succeed_rate = (self.build_calc_succeed_rate_dataset(x[0], original_y, data_preprocess, steps_done, True),)
            outs = outs + succeed_rate
            self.model.train()
        return self.run_after_finished_batch(outs, batch_logs, callbacks, batch_index, steps_done)

    def train_on_batch(self, x, y, sample_weight=None, data_preprocess=None):
        self.model.to(self.torch_device)
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.convert_data_for_model(x, y)
        outputs = self.model(x)
        if self.is_siamese_inceptionV3:
            loss, aux_loss = self.loss(outputs, y)
            loss.backward(retain_graph=True)
            running_loss = loss.item()
            aux_loss.backward(retain_graph=True)
            aux_running_loss = aux_loss.item()
            self.optimizer.step()
            predicted, aux_predicted = self.get_predicted(outputs)
            collect_rate = self.calc_collect_rate(predicted, y[0])
            aux_collect_rate = self.calc_collect_rate(aux_predicted, y[1])
            return running_loss, collect_rate, aux_running_loss, aux_collect_rate
        if self.is_inceptionV3:
            loss = self.loss(outputs.logits, y)
            self.optimizer.step()
            running_loss = loss.item()
            predicted = self.get_predicted(outputs.logits)
            if outputs.aux_logits is not None:
                aux_loss = self.__loss(outputs.aux_logits, y)
                self.optimizer.step()
            return running_loss, self.calc_collect_rate(predicted, y)

    def add_output_param_to_batch_log_param(self, outs, batch_logs):
        batch_logs["loss"] = outs[0]
        batch_logs["accuracy"] = outs[1]
        batch_logs["aux_loss"] = outs[2]
        batch_logs["aux_accuracy"] = outs[3]
        batch_logs["original_accuracy"] = outs[4][0]
        batch_logs["aux_original_accuracy"] = outs[4][1]
        return batch_logs
