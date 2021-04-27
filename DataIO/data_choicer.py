import os
import glob
from typing import TypeVar, List, Tuple, Union
from random import choices
from numba import jit
import shutil

T = TypeVar('T')
ChoiceDataNum = Union[int, float]


class BaggingDataPicker:

    def __init__(self, pick_data_num: Union[ChoiceDataNum, List[ChoiceDataNum]], build_dataset_num: int = 5):
        self.__pick_data_num = pick_data_num
        self.__build_dataset_num = len(pick_data_num) if type(pick_data_num) is list else build_dataset_num

    @property
    def pick_data_num(self) -> Union[int, List[int]]:
        return self.__pick_data_num

    @property
    def build_dataset_num(self) -> int:
        return self.__build_dataset_num

    def pickup_dataset(self, data_set: List[T]) -> List[List[T]]:
        if type(self.pick_data_num) is list:
            return [choice_dataset(data_set, pick_data_num) for pick_data_num in self.pick_data_num]
        return [choice_dataset(data_set, self.pick_data_num) for _ in range(self.build_dataset_num)]

    def pickup_dataset_from_dir(self, dataset_dir_path: str) -> List[List[T]]:
        dataset = glob.glob(os.path.join(dataset_dir_path, '*.jpg'))
        return self.pickup_dataset(dataset)

    def pickup_dataset_from_data_dir(self, dataset_dir_path: str) -> Tuple[List[str], List[List[List[str]]]]:
        class_set = os.listdir(dataset_dir_path)
        class_data_paths = [os.path.join(dataset_dir_path, class_name) for class_name in class_set]
        pickup_datasets = [self.pickup_dataset_from_dir(data_class_path) for data_class_path in class_data_paths]
        return class_set, pickup_datasets

    def copy_dataset_for_bagging(self, dataset_dir_path: str) -> str:
        bagging_dir = os.path.join(dataset_dir_path, 'bagging')
        class_sets, pickup_data_sets = self.pickup_dataset_from_data_dir(dataset_dir_path)
        for class_name, data_sets in zip(class_sets, pickup_data_sets):
            copy_datasets(bagging_dir, class_name, data_sets)
        return bagging_dir


@jit
def choice_dataset(data_set: List[T], choice_data_param: ChoiceDataNum) -> List[T]:
    pick_data_num = int(len(data_set)*choice_data_param) if type(choice_data_param) is float else choice_data_param
    return choices(data_set, k=pick_data_num)


@jit
def copy_datasets(bagging_dir: str, class_name: str, data_sets: List[List[str]]):
    for index, data_set in enumerate(data_sets):
        write_dir_path = os.path.join(bagging_dir, str(index), class_name)
        if os.path.exists(write_dir_path) is False:
            print("build dir", write_dir_path)
            os.makedirs(write_dir_path)
        copy_dataset(data_set, write_dir_path)


@jit
def copy_dataset(original_paths: List[str], target_dir: str):
    for index, original_path in enumerate(original_paths):
        copy_data(index, original_path, target_dir)


@jit
def copy_data(index: int, original_path: str, target_dir: str):
    write_path = os.path.join(target_dir, str(index) + '.jpg')
    shutil.copyfile(original_path, write_path)