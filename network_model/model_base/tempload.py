import keras.engine.training
from tensorflow.keras.optimizers import Optimizer, SGD
from typing import Union, Tuple, Optional, List
from keras.models import load_model, model_from_json, model_from_yaml
import json
import yaml
import os


def load_model_arc(model_arc_path: str) -> keras.engine.training.Model:
    _, ext = os.path.splitext(model_arc_path)
    is_yaml = ext == '.yaml' or ext == '.yml'

    def build_from_yaml(model_str: str) -> keras.engine.training.Model:
        return model_from_yaml(model_str)

    def build_from_json(model_str: str) -> keras.engine.training.Model:
        return model_from_json(model_str)

    build_model = build_from_yaml if is_yaml else build_from_json
    with open(model_arc_path, 'r') as f:
        raw_str = f.read()
        return build_model(raw_str)


class TempLoader:

    def __init__(self,
                 loss: str = "categorical_crossentropy",
                 metrics: Optional[List[str]] = None,
                 will_show_summary: bool = True):
        if metrics is None:
            metrics = ['accuracy']
        self.__loss = loss
        self.__metrics = metrics
        self.__will_show_summary = will_show_summary

    def load_simple_model(self, temp_filepath: str, optimizer: Optimizer = SGD()):
        model = load_model(temp_filepath)
        # モデルの概要を表示
        if self.__will_show_summary:
            model.summary()

        # モデルをコンパイル
        model.compile(loss=self.__loss, optimizer=optimizer, metrics=self.__metrics)
        return model

    def load_with_weight_file(self, model_arc_path: str, model_weight_path: str, optimizer: Optimizer = SGD()):
        model = load_model_arc(model_arc_path)
        print(model)
        model.load_weights(model_weight_path)
        model.compile(loss=self.__loss, optimizer=optimizer, metrics=self.__metrics)
        if self.__will_show_summary:
            model.summary()
        return model

    def __call__(self, temp_paths: Union[str, Tuple[str, str]], optimizer: Optimizer = SGD()):
        print(temp_paths)
        if type(temp_paths) is str:
            return self.load_simple_model(temp_paths, optimizer)
        model_arc_path, model_weight_path = temp_paths
        return self.load_with_weight_file(model_arc_path, model_weight_path, optimizer)


builder = TempLoader()
builder_for_merge = TempLoader(will_show_summary=False)
