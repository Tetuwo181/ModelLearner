import keras.engine.training
from keras.layers import Input, Dense
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot
from keras.optimizers import Optimizer, SGD
from keras.models import Model
from tensorflow.python.framework.ops import Tensor
from typing import List, Union, Optional, Callable, Tuple
from network_model.model_base.tempload import builder_for_merge

Merge = Union[Add, Subtract, Multiply, Average, Minimum, Maximum, Concatenate, Dot]
Loss = Union[str, Callable[[Tensor, Tensor], Tensor]]


class ModelMerger:

    def __init__(self,
                 merge_obj: Merge,
                 loss: Loss = "categorical_crossentropy",
                 optimizer: Optimizer = SGD(),
                 metrics: Optional[List[str]] = None,
                 output_activation: str = "softmax"):
        if metrics is None:
            metrics = ['accuracy']
        self.__loss = loss
        self.__optimizer = optimizer
        self.__metrics = metrics
        self.__merge_obj = merge_obj
        self.__output_activation = output_activation

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @property
    def metrics(self) -> List[str]:
        return self.__metrics

    @property
    def merge(self) -> Merge:
        return self.__merge_obj

    @property
    def loss(self) -> Loss:
        return self.__loss

    @property
    def output_activation(self):
        return self.__output_activation

    def get_output_num(self, model: keras.engine.training.Model, output_num: Optional[int] = None):
        if output_num is None:
            return model.output_shape[-1]
        if type(self.merge) is Concatenate:
            return output_num
        return model.output_shape[-1]

    def merge_models(self,
                     models: List[keras.engine.training.Model],
                     output_num: Optional[int] = None,
                     middle_layer_neuro_nums: Optional[List[Tuple[int, str]]] = None) -> keras.engine.training.Model:
        input_shape = tuple(models[0].input_shape[1:])
        print(input_shape)
        output_class_num = models[0].output_shape[-1] if output_num is None else output_num
        print(output_class_num)
        input_layer = Input(shape=input_shape)
        model_outputs = [model(input_layer) for model in models]
        added_model_output = self.merge(model_outputs)
        if middle_layer_neuro_nums is None:
            output = Dense(output_class_num, activation=self.output_activation)(added_model_output)
            model = Model(input_layer, output)
            # モデルの概要を表示
            model.summary()

            # モデルをコンパイル
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            return model
        add_layers = middle_layer_neuro_nums + [(output_class_num, self.output_activation)]
        print(add_layers)
        output = Dense(add_layers[0][0], activation=add_layers[0][1])(added_model_output)
        for params in add_layers[1:]:
            print(params)
            output = Dense(params[0], activation=params[1])(output)
        model = Model(input_layer, output)
        model.summary()
        # モデルをコンパイル
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def merge_models_from_model_files(self,
                                      h5_paths: List[Union[str, Tuple[str, str]]],
                                      trainable_model: Union[bool, List[bool]] = True,
                                      output_num: Optional[int] = None,
                                      middle_layer_neuro_nums: Optional[List[Tuple[int, str]]] = None,
                                      merge_per_model_name: str = 'model') -> keras.engine.training.Model:
        models = [builder_for_merge(h5_path) for h5_path in h5_paths]
        are_trainable_models = [trainable_model for _ in h5_paths] if type(trainable_model) is bool else trainable_model
        for index, (model, is_trainable) in enumerate(zip(models, are_trainable_models)):
            model._name = merge_per_model_name + str(index)
            model.trainable = is_trainable
        return self.merge_models(models, output_num, middle_layer_neuro_nums)


