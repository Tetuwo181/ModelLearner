
import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from keras.models import load_model
from typing import Callable
import tensorflow as tf


def pb_builder(output_dir: str) -> Callable[[Model, str, str], None]:
    def convert_2_pb(base_model: Model,
                     output_name: str,
                     output_node_name: str) -> None:
        input_shape = tuple([1] + base_model.input_shape[1:])
        x = tf.placeholder(tf.float32, input_shape, name="model_input")
        y = base_model(x)
        base_model.summary()
        uninitialized_variables = [v for v in tf.global_variables() \
                                   if not hasattr(v, '_keras_initialized') or not v._keras_initialized]
        sess = K.get_session()

        gd = sess.graph.as_graph_def()
        sess.run(tf.variables_initializer(uninitialized_variables))
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        gd,
                                                                        [output_node_name])
        tf.train.write_graph(frozen_graph_def,
                             output_dir,
                             name=output_name,
                             as_text=False)
    return convert_2_pb


def build_from_model_path(base_model_path: str,
                          output_dir: str):
    base_model = load_model(base_model_path)
    builder = pb_builder(output_dir)

    def convert_2_pb(output_name: str, output_node_name: str):
        builder(base_model, output_name, output_node_name)

    return convert_2_pb
