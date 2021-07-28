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
        super(AbstractKerasWrapper, self).__init__(shape,
                                                   class_set,
                                                   callbacks,
                                                   monitor,
                                                   preprocess_for_model,
                                                   after_learned_process)

    @property
    def will_save_h5(self):
        return self.__will_save_h5
