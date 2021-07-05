from model_merger.type import Merge, Loss, TrainableModelIndex
from model_merger.proc.calculator import LCaliculator
from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np


class ShameBuilder(object):

    def __init__(self, base_model: Model):
        self.__base_model = base_model

    @property
    def output_shape(self):
        return self.__base_model.output_shape

    @property
    def output_layer(self):
        return self.__base_model.output

    @property
    def teacher_input_layer(self):
        return Input(shape=tuple(self.__base_model.output_shape[1:]))

    @property
    def input_layer(self):
        return Input(shape=tuple(self.__base_model.input_shape[1:]))

    def build_input_teacher_from_output(self):
        return Input(shape=self.output_shape)

    def build_shame_trainer_for_classifivation(self,
                                               calculator: LCaliculator,
                                               optimizer):
        input_for_batch_data = [self.input_layer, self.input_layer]
        predict_outputs = [self.__base_model(input_batch) for input_batch in input_for_batch_data]
        teacher_inputs = [self.teacher_input_layer, self.teacher_input_layer]
        loss_inputs = predict_outputs + teacher_inputs
        output_loss = calculator.build_loss_layer()(loss_inputs)
        inputs = input_for_batch_data + teacher_inputs
        train_model = Model(inputs=inputs, outputs=output_loss)
        train_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
        return train_model


