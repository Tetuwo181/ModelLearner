from model_merger.type import Merge, Loss, TrainableModelIndex
from keras.models import Model
from keras.layers import Input
import numpy as np
from math import e as exponential


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
    def input_layer(self):
        return self.__base_model.inputs

    def build_input_teacher_from_output(self):
        return Input(shape=self.output_shape)

    def build_shame_trainer_for_classifivation(self):
        input_for_batch_data = [self.input_layer, self.input_layer]
        output_for_batch_data = [self.output_layer, self.input_layer]


class LCaliculator(object):

    def __init__(self, q):
        """

        :param q:　https://www.mdpi.com/2073-8994/10/9/385の論文に掲載されているQの値のパラメータ
        """

        self.__q = q

    @property
    def q(self):
        return self.__q

    def l_minus(self, x):
        return 2*np.power(self.q*exponential, -((2.77/self.q)*x))

    def l_plus(self, x):
        return (2/self.q)*np.power(x, 2)


def calc_l1_norm(output_base, output_other):
    return np.linalg.norm(output_base-output_other, 1)


def is_same_class(base_classes, other_classes):
    return np.linalg.norm(base_classes-other_classes, 1) == 0
