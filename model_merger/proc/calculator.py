from math import e as exponential
import numpy as np
from keras.layers import Lambda
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend as K


class LCaliculator(object):

    def __init__(self, q = 1):
        """

        :param q:　https://www.mdpi.com/2073-8994/10/9/385の論文に掲載されているQの値のパラメータ
        """

        self.__q = q

    @property
    def q(self):
        return self.__q

    @property
    def loss_func(self):
        def loss(y_true, y_pred):
            return K.mean(y_true*K.square(y_pred) + (1 - y_true)*K.square(K.maximum(self.__q - y_pred, 0)))
        return loss

    def l_minus(self, x):
        return 2*tf.math.pow(self.q*exponential, -((2.77/self.q)*x))

    def l_plus(self, x):
        return (2/self.q)*tf.math.pow(x, 2)

    def build_loss_layer(self, name="kd_"):
        return Lambda(_eucl_dist_output_shape, name=name, output_shape=_eucl_dist_output_shape)


def calc_l1_norm(vects):
    output_base, output_other = vects
    return K.sqrt(K.sum(K.square(output_base-output_other), axis=1, keepdims=True))


def _eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true*K.square(y_pred) + (1 - y_true)*K.square(K.maximum(margin - y_pred, 0)))