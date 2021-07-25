from math import e as exponential
import numpy as np
from keras.layers import Lambda
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend as K


def calc_l1_norm(vects):
    output_base, output_other = vects
    return K.sum(output_base-output_other, axis=1, keepdims=True)


def calc_l2_norm(vects):
    output_base, output_other = vects
    return K.sqrt(K.sum(K.square(output_base-output_other), axis=1, keepdims=True))


def loss_func_mean(y_true, y_pred):
    return K.mean(y_true*K.square(y_pred) + (1 - y_true)*K.square(K.maximum(1.0 - y_pred, 0)))


def _eucl_dist_output_shape(shapes):
    print(shapes, type(shapes))
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true*K.square(y_pred) + (1 - y_true)*K.square(K.maximum(margin - y_pred, 0)))


class LCaliculator(object):

    def __init__(self,
                 q=100,
                 loss_func=None,
                 calc_distance=calc_l1_norm):
        """

        :param q:　https://www.mdpi.com/2073-8994/10/9/385の論文に掲載されているQの値のパラメータ
        """

        self.__q = q
        self.__loss_func = loss_func
        self.__calc_distance = calc_distance

    @property
    def q(self):
        return self.__q

    @property
    def default_loss_func(self):
        def loss(y_true, y_pred):
            return y_true*self.l_plus(y_pred) + (1-y_true)*self.l_minus(y_pred)
        return loss

    @property
    def loss_func(self):
        return self.default_loss_func if self.__loss_func is None else self.__loss_func

    def l_minus(self, x):
        return 2*K.pow(self.q*exponential, -((2.77/self.q)*x))

    def l_plus(self, x):
        return (2/self.q)*K.square(x)

    def build_loss_layer(self, name="kd_"):
        return Lambda(self.__calc_distance, name=name, output_shape=_eucl_dist_output_shape)


