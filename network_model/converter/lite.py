import tensorflow as tf
import os
from typing import Union, List


def build_base_converter(input_model_path: str):
    keras_model = tf.keras.models.load_model(input_model_path)
    return tf.lite.TFLiteConverter.from_keras_model(keras_model)


def convert_tflite_default(input_model_path: str):
    converter = build_base_converter(input_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_tflite_int8_model(input_model_path: str):
    converter = build_base_converter(input_model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    return converter.convert()


def convert_tflite_float16_model(input_model_path: str):
    converter = build_base_converter(input_model_path)
    converter.target_spec.supported_ops = [tf.compat.v1.lite.constants.FLOAT16]

    return converter.convert()


def get_tflite_converter(mode: str = "default"):
    if mode == "int8":
        return convert_tflite_int8_model
    if mode == "float16":
        return convert_tflite_float16_model
    return convert_tflite_default


def keras_2_tflite(input_model_path: str, output_path: str, mode: str = "default") -> None:
    converter = get_tflite_converter(mode)
    tflite_model = converter(input_model_path)
    with open(output_path, 'wb') as o_:
        o_.write(tflite_model)


def build_after_record_tflite_recorder(input_model_path: str, mode: Union[str, List[str]] = "default"):
    """
    学習を行った後モデルを返還するために必要なラッパー
    :param input_model_path:
    :param mode:
    :return:
    """
    output_dir_path = os.path.dirname(input_model_path)
    file_name_base = os.path.splitext(os.path.basename(input_model_path))[0]

    def build_tflite():
        if type(mode) is str and mode != "all":
            output_path = os.path.join(output_dir_path, file_name_base + "_" + mode + ".tflite")
            return keras_2_tflite(input_model_path, output_path, mode)
        write_modes = ["int8", "float16", "default"] if mode == "all" else mode
        for write_mode in write_modes:
            output_path = os.path.join(output_dir_path, file_name_base + "_" + write_mode + ".tflite")
            keras_2_tflite(input_model_path, output_path, write_mode)
    return build_tflite
