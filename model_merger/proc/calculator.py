from math import e as exponential
import numpy as np
from keras.layers import Lambda
import tensorflow as tf
from keras.callbacks import Callback


class LCaliculator(object):

    def __init__(self, q):
        """

        :param q:　https://www.mdpi.com/2073-8994/10/9/385の論文に掲載されているQの値のパラメータ
        """

        self.__q = q

    def __call__(self, inputs):
        base_output, other_output, base_teacher, other_teacher = inputs
        distance = calc_l1_norm(base_output, other_output)
        return self.l_minus(distance) if is_same_class(base_teacher, other_teacher) else self.l_plus(distance)

    @property
    def q(self):
        return self.__q

    @property
    def loss_func(self):
        @tf.function
        def loss(inputs):
            base_output, other_output, base_teacher, other_teacher = inputs
            distance = calc_l1_norm(base_output, other_output)
            return self.l_minus(distance) if is_same_class(base_teacher, other_teacher) else self.l_plus(distance)
        return loss

    @tf.function
    def l_minus(self, x):
        return 2*tf.math.pow(self.q*exponential, -((2.77/self.q)*x))

    @tf.function
    def l_plus(self, x):
        return (2/self.q)*tf.math.pow(x, 2)

    def build_loss_layer(self, name="kd_"):
        return Lambda(self.loss_func, name=name, output_shape=(1,))


def calc_l1_norm(output_base, output_other):
    return tf.norm(output_base-output_other, 1)


def is_same_class(base_classes, other_classes):
    return tf.norm(base_classes-other_classes, 1) == 0
