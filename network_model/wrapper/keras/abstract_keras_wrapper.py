import keras.callbacks
from typing import List
from typing import Optional
from typing import Callable
from network_model.wrapper.abstract_model import AbstractModel
from abc import ABC
ModelPreProcessor = Optional[Callable[[keras.engine.training.Model],  keras.engine.training.Model]]


class AbstractKerasWrapper(AbstractModel, ABC):
    def __init__(self,
                 shape,
                 class_set: List[str],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 monitor: str = "",
                 will_save_h5: bool = True,
                 preprocess_for_model: ModelPreProcessor = None,
                 after_learned_process: Optional[Callable[[None], None]] = None):
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
        return model_name + ".h5" if self.will_save_h5 else model_name

    def build_best_model_file_name(self, model_name):
        return model_name + "_best.h5" if self.will_save_h5 else model_name + "_best"