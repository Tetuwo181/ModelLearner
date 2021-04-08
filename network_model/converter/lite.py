import numpy as np
import tensorflow as tf


def convert_2_tflite(input_model_path: str,
                     input_name: str,
                     output_path: str,
                     output_node_name: str) -> None:
    converter = tf.lite.TFLiteConverter.from_frozen_graph(input_model_path,
                                                          [input_name],
                                                          [output_node_name])
    tflite_quant_model = converter.convert()
    with open(output_path, 'wb') as o_:
        o_.write(tflite_quant_model)

