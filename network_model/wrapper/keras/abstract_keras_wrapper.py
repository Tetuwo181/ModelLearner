from __future__ import annotations
from tensorflow.keras import Model
import keras.callbacks
from typing import List
from typing import Optional
from typing import Callable
from network_model.wrapper.abstract_model import AbstractModel
from abc import ABC
ModelPreProcessor = Callable[[Model],  Model] | None


class AbstractKerasWrapper(AbstractModel, ABC):
    def __init__(self,
                 shape,
                 class_set: list[str],
                 callbacks: list[keras.callbacks.Callback] | None = None,
                 monitor: str = "",
                 will_save_h5: bool = True,
                 preprocess_for_model: ModelPreProcessor = None,
                 after_learned_process: Callable[[None], None] | None = None):
        self.__will_save_h5 = will_save_h5
        self.__input_shape = shape
        super(AbstractKerasWrapper, self).__init__(
                                                   class_set,
                                                   callbacks,
                                                   monitor,
                                                   preprocess_for_model,
                                                   after_learned_process)

    @property
    def will_save_h5(self):
        return self.__will_save_h5

    @property
    def input_shape(self):
        return self.__input_shape

    def build_write_set(self):
        return {"class_set": self.class_set, "input_shape": self.input_shape}

    def build_model_file_name(self, model_name):
        return model_name + ".keras" if self.will_save_h5 else model_name

    def build_best_model_file_name(self, model_name):
        return model_name + "_best.keras" if self.will_save_h5 else model_name + "_best"

    def save_model(self, file_path):
        self.model.save(file_path)

    def build_model_checkpoint(self, temp_best_path, save_weights_only):
        return keras.callbacks.ModelCheckpoint(temp_best_path,
                                               monitor=self.monitor,
                                               save_best_only=True,
                                               save_weights_only=save_weights_only)

