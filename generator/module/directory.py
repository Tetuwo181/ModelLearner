from keras_preprocessing.image.directory_iterator import DirectoryIterator
from typing import Callable, Optional
import numpy as np


class DirectoryIteratorWithPreprocess(DirectoryIterator):

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 x_preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 y_preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        super().__init__(directory,
                         image_data_generator,
                         target_size,
                         color_mode,
                         classes,
                         class_mode,
                         batch_size,
                         shuffle,
                         seed,
                         data_format,
                         save_to_dir,
                         save_prefix,
                         save_format,
                         follow_links,
                         subset,
                         interpolation,
                         dtype)
        self.__x_preprocess = x_preprocess
        self.__y_preprocess = y_preprocess

    @property
    def x_preprocess(self):
        return self.__x_preprocess

    @property
    def y_preprocess(self):
        return self.__y_preprocess

    def next(self):
        base_params = super().next()
        batch_x = base_params[0] if self.x_preprocess is None else self.x_preprocess(base_params[0])
        batch_y = base_params[1] if self.y_preprocess is None else self.y_preprocess(base_params[1])
        if len(base_params) == 2:
            return batch_x, batch_y
        return batch_x, batch_y, base_params[3]
