import keras.engine.training
from typing import Callable


def fix_layer_weight(layer_depth: int) -> Callable[[keras.engine.training.Model], keras.engine.training.Model]:
    def fix_weight(target_model: keras.engine.training.Model):
        for layer in target_model.layers[:layer_depth]:
            layer.trainable = False
        return target_model
    return fix_weight
