import tensorflow as tf


def keras_2_tflite(input_model_path: str, output_path: str) -> None:
    keras_model = tf.keras.models.load_model(input_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    print(converter)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as o_:
        o_.write(tflite_model)

