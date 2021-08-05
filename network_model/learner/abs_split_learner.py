import os
import shutil
import random
from typing import Tuple, List, Union, Callable
from typing import Optional
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator
from network_model.wrapper.keras import many_data as md
from network_model.model_builder import ModelBuilder
from DataIO.data_loader import count_data_num_in_dir
from DataIO.data_loader import NormalizeType
from DataIO.data_choicer import BaggingDataPicker, ChoiceDataNum
from abc import ABC
from network_model.model_for_distillation import ModelForDistillation
from network_model.builder.pytorch_builder import PytorchModelBuilder


LearnModel = Union[md.ModelForManyData, ModelForDistillation]


def image_dir_train_test_split(original_dir, base_dir, train_size=0.8, has_built: bool = True):
    '''
    画像データをトレインデータとテストデータにシャッフルして分割
    下記のURLで公開されていたコードの改変です
    https://qiita.com/komiya-m/items/c37c9bc308d5294d3260

    parameter
    ------------
    original_dir: str オリジナルデータフォルダのパス その下に各クラスのフォルダがある
    base_dir: str 分けたデータを格納するフォルダのパス　そこにフォルダが作られます
    train_size: float トレインデータの割合
    '''
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print(base_dir + "は作成済み")

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir)
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))]
    original_dir_path = [os.path.join(original_dir, p) for p in dir_lists]

    if has_built:
        return dir_lists, \
               count_data_num_in_dir(os.path.join(base_dir, 'train')), \
               count_data_num_in_dir(os.path.join(base_dir, 'validation')), \
               count_data_num_in_dir(original_dir)

    num_class = len(dir_lists)

    # フォルダの作成(トレインとバリデーション)
    train_dir = os.path.join(base_dir, 'train')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, 'validation')
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.mkdir(validation_dir)

    #クラスフォルダの作成
    train_dir_path_lists = []
    val_dir_path_lists = []
    for directory_name in dir_lists:
        train_class_dir_path = os.path.join(train_dir, directory_name)
        os.mkdir(train_class_dir_path)
        train_dir_path_lists += [train_class_dir_path]

        val_class_dir_path = os.path.join(validation_dir, directory_name)
        os.mkdir(val_class_dir_path)
        val_dir_path_lists += [val_class_dir_path]


    #元データをシャッフルしたものを上で作ったフォルダにコピーします。
    #ファイル名を取得してシャッフル
    for i, path in enumerate(original_dir_path):
        files_class = os.listdir(path)
        random.shuffle(files_class)
        # 分割地点のインデックスを取得
        divide_num = int(len(files_class) * train_size)
        #トレインへファイルをコピー
        for file_name in files_class[:divide_num]:
            src = os.path.join(path, file_name)
            dst = os.path.join(train_dir_path_lists[i], file_name)
            shutil.copyfile(src, dst)
        #valへファイルをコピー
        for file_name in files_class[divide_num:]:
            src = os.path.join(path, file_name)
            dst = os.path.join(val_dir_path_lists[i], file_name)
            shutil.copyfile(src, dst)
        print(path + "コピー完了")

    print("分割終了")
    return dir_lists, \
           count_data_num_in_dir(os.path.join(base_dir, 'train')), \
           count_data_num_in_dir(os.path.join(base_dir, 'validation')), \
           count_data_num_in_dir(original_dir)


class AbsModelLearner(ABC):

    def __init__(self,
                 model_builder: Union[ModelBuilder, PytorchModelBuilder],
                 train_image_generator: ImageDataGenerator,
                 test_image_generator: ImageDataGenerator,
                 class_list: List[str],
                 normalize_type: NormalizeType = NormalizeType.Div255,
                 callbacks: Optional[List[keras.callbacks.Callback]] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 train_dir_name: str = "train",
                 validation_name: str = "validation",
                 will_save_h5: bool = True,
                 preprocess_for_model= None,
                 after_learned_process: Optional[Callable[[None], None]] = None,
                 class_mode: Optional[str] = None):
        """

        :param model_builder: モデル生成器
        :param train_image_generator: 教師データ生成ジェネレータ
        :param test_image_generator: テストデータ生成ジェネレータ
        :param class_list: 学習対象となるクラス
        :param normalize_type: 正規化の手法
        :param callbacks: kerasで学習させる際のコールバック関数
        :param image_size: 画像の入力サイズ
        :param train_dir_name: 検証する際の教師データのディレクトリ名
        :param validation_name: 検証する際のテストデータのディレクトリ名
        :param will_save_h5: 途中モデル読み込み時に旧式のh5ファイルで保存するかどうか　デフォルトだと保存する
        :param preprocess_for_model: メインの学習前にモデルに対して行う前処理
        :param after_learned_process: モデル学習後の後始末
        :param class_mode: flow_from_directoryのクラスモード
        """

        self.__model_builder = model_builder
        self.__normalize_type = normalize_type
        self.__train_image_generator = train_image_generator
        self.__test_image_generator = test_image_generator
        self.__callbacks = callbacks
        self.__class_list = class_list
        self.__image_size = image_size
        self.__train_dir_name = train_dir_name
        self.__validation_name = validation_name
        self.__will_save_h5 = will_save_h5
        self.__preprocess_for_model = preprocess_for_model
        self.__after_learned_process = after_learned_process
        self.__class_mode = class_mode

    @property
    def preprocess_for_model(self):
        return self.__preprocess_for_model

    @property
    def after_learned_process(self):
        return self.__after_learned_process

    @property
    def model_builder(self) -> ModelBuilder:
        """

        :return: モデル生成器
        """
        return self.__model_builder

    @property
    def callbacks(self) -> Optional[List[keras.callbacks.Callback]]:
        """

        :return: 学習させる際のコールバック関数
        """
        return self.__callbacks

    @property
    def class_list(self):
        return self.__class_list

    @property
    def class_num(self) -> int:
        """

        :return: 学習させる際のクラス数
        """
        return len(self.class_list)

    @property
    def image_size(self) -> Tuple[int, int]:
        """

        :return: 入力画像サイズ
        """
        return self.__image_size

    @property
    def normalize_type(self):
        return self.__normalize_type

    @property
    def will_save_h5(self):
        return self.__will_save_h5

    @property
    def train_dir_name(self):
        return self.__train_dir_name

    @property
    def validation_name(self):
        return self.__validation_name

    @property
    def class_mode(self):
        if self.__class_mode is not None:
            return self.__class_mode
        if self.class_num > 2:
            return "categorical"
        return "binary"

    @property
    def is_torch(self):
        return isinstance(self.__model_builder, PytorchModelBuilder)

    @staticmethod
    def get_train_and_test_num(base_dir: str):
        return count_data_num_in_dir(os.path.join(base_dir, 'train')), \
               count_data_num_in_dir(os.path.join(base_dir, 'validation'))

    def build_train_generator(self, batch_size, train_dir: str):
        return self.__train_image_generator.flow_from_directory(train_dir,
                                                                target_size=self.image_size,
                                                                batch_size=batch_size,
                                                                classes=self.class_list,
                                                                class_mode=self.class_mode)

    def build_test_generator(self, batch_size, test_data_dir: str):
        return self.__test_image_generator.flow_from_directory(test_data_dir,
                                                               target_size=self.image_size,
                                                               batch_size=batch_size,
                                                               classes=self.class_list,
                                                               class_mode=self.class_mode)

    def build_model(self,
                    model_dir_path: str,
                    result_name: str,
                    tmp_model_path: str = None,
                    monitor: str = "") -> LearnModel:
        pass

    def build_train_validation_dir_paths(self, base_dir: str) -> Tuple[str, str]:
        return os.path.join(base_dir, self.train_dir_name), os.path.join(base_dir, self.validation_name)

    def train_with_validation_from_model(self,
                                         model: LearnModel,
                                         result_dir_path: str,
                                         train_dir: str,
                                         validation_dir: str,
                                         batch_size=32,
                                         epoch_num: int = 20,
                                         result_name: str = "result",
                                         model_name: str = "model",
                                         save_weights_only: bool = False,
                                         will_use_multi_inputs_per_one_image: bool = False,
                                         input_data_preprocess_for_building_multi_data=None) -> LearnModel:
        train_generator, train_steps_per_epoch, test_generator, test_steps_per_epoch = \
            self.build_validation_generator_and_get_steps_per_epoch(train_dir, validation_dir, batch_size)

        # テスト開始
        model.test(train_generator,
                   epoch_num,
                   test_generator,
                   normalize_type=self.normalize_type,
                   result_dir_name=result_name,
                   steps_per_epoch=train_steps_per_epoch,
                   validation_steps=test_steps_per_epoch,
                   dir_path=result_dir_path,
                   model_name=result_name+"val",
                   save_weights_only=save_weights_only,
                   will_use_multi_inputs_per_one_image=will_use_multi_inputs_per_one_image,
                   input_data_preprocess_for_building_multi_data=input_data_preprocess_for_building_multi_data
                   )
        return model

    def train_with_validation(self,
                              dataset_root_dir: str,
                              result_dir_path: str,
                              batch_size=32,
                              epoch_num: int = 20,
                              result_name: str = "result",
                              model_name: str = "model",
                              tmp_model_path: str = None,
                              monitor: str = "",
                              save_weights_only: bool = False,
                              will_use_multi_inputs_per_one_image: bool = False,
                              data_preprocess=None) -> LearnModel:
        """
        検証用データがある場合の学習
        :param dataset_root_dir: データが格納されたディレクトリ
        :param result_dir_path: モデルを出力するディレクトリ
        :param batch_size: 学習する際のバッチサイズ
        :param epoch_num: 学習する際のエポック数
        :param result_name: 出力する結果名
        :param model_name: モデル名
        :param tmp_model_path: 学習済みのh5ファイルからモデルを読み込んで学習する際のh5ファイルのパス
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param save_weights_only:
        :param will_use_multi_inputs_per_one_image:
        :param data_preprocess:
        :return: 学習済みモデル
        """
        model_val = self.build_model(tmp_model_path, monitor)
        train_dir, validation_dir = self.build_train_validation_dir_paths(dataset_root_dir)
        return self.train_with_validation_from_model(model_val,
                                                     result_dir_path,
                                                     train_dir,
                                                     validation_dir,
                                                     batch_size,
                                                     epoch_num,
                                                     result_name,
                                                     model_name,
                                                     save_weights_only,
                                                     will_use_multi_inputs_per_one_image,
                                                     data_preprocess)

    def train_without_validation(self,
                                 original_dir: str,
                                 result_dir_path: str,
                                 batch_size=32,
                                 epoch_num: int = 20,
                                 result_name: str = "result",
                                 model_name: str = "model",
                                 tmp_model_path: str = None,
                                 monitor: str = "",
                                 save_weights_only: bool = False,
                                 will_use_multi_inputs_per_one_image: bool = False,
                                 data_preprocess=None) -> LearnModel:
        model = self.build_model(result_dir_path, result_name, tmp_model_path, monitor)
        train_generator = self.build_train_generator(batch_size, original_dir)
        data_num = count_data_num_in_dir(original_dir)
        model.fit_generator(train_generator,
                            epoch_num,
                            steps_per_epoch=(data_num//batch_size),
                            save_weights_only=save_weights_only,
                            will_use_multi_inputs_per_one_image=will_use_multi_inputs_per_one_image,
                            data_preprocess=data_preprocess)
        model.record(result_name,
                     result_dir_path,
                     model_name,
                     normalize_type=self.normalize_type)
        return model

    def cross_validation_from_pre_splitted(self,
                                           base_dir: str,
                                           result_dir_path: str,
                                           batch_size=32,
                                           epoch_num: int = 20,
                                           result_name: str = "result",
                                           model_name: str = "model",
                                           tmp_model_path: str = None,
                                           start_index: Optional[int] = None,
                                           monitor: str = None,
                                           save_weights_only: bool = False,
                                           will_use_multi_inputs_per_one_image: bool = False,
                                           data_preprocess=None) -> List[LearnModel]:
        """
        あらかじめ交差検証のためにデータ分割したディレクトリから検証を行い学習する
        :param base_dir: 検証データが格納されたディレクトリ
        :param result_dir_path: モデルを出力するディレクトリ
        :param batch_size: 学習する際のバッチサイズ
        :param epoch_num: 学習する際のエポック数
        :param result_name: 出力する結果名
        :param model_name: モデル名
        :param tmp_model_path: 学習済みのh5ファイルからモデルを読み込んで学習する際のh5ファイルのパス
        :param start_index: 途中から始める場合のインデックス 指定しない場合は初めからする
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param save_weights_only:
        :param will_use_multi_inputs_per_one_image:
        :param data_preprocess:
        :return:
        """
        val_dir_names_base = os.listdir(base_dir)
        val_dir_names = val_dir_names_base if start_index is None else val_dir_names_base[start_index:]
        result_names = [result_name + dir_name for dir_name in val_dir_names]
        model_names = [model_name + dir_name for dir_name in val_dir_names]
        data_dir_paths = [os.path.join(base_dir, val_dir_name) for val_dir_name in val_dir_names]
        models = [self.train_with_validation(data_dir_path,
                                             result_dir_path,
                                             batch_size,
                                             epoch_num,
                                             result_name,
                                             model_name,
                                             tmp_model_path,
                                             monitor,
                                             save_weights_only,
                                             will_use_multi_inputs_per_one_image,
                                             data_preprocess)
                  for result_name, model_name, data_dir_path in zip(result_names, model_names, data_dir_paths)]
        return models

    def train_by_bagging(self,
                         dataset_root_dir: str,
                         result_dir_path: str,
                         pick_data_num: Union[ChoiceDataNum, List[ChoiceDataNum]],
                         build_dataset_num: int = 5,
                         batch_size=32,
                         epoch_num: int = 20,
                         result_name: str = "result",
                         model_name: str = "model",
                         tmp_model_path: str = None,
                         monitor: str = "",
                         save_weights_only: bool = False,
                         will_use_multi_inputs_per_one_image: bool = False,
                         data_preprocess=None) -> List[LearnModel]:
        """
        バギングで学習する
        :param dataset_root_dir: データが格納されたディレクトリ
        :param result_dir_path: モデルを出力するディレクトリ
        :param pick_data_num: 1クラスあたり抽出するデータ数 リストで渡すとそのリストの中に格納された各値だけデータを抽出したデータセットを作成する
        :param build_dataset_num: データセットを作る数
        :param batch_size: 学習する際のバッチサイズ
        :param epoch_num: 学習する際のエポック数
        :param result_name: 出力する結果名
        :param model_name: モデル名
        :param tmp_model_path: 学習済みのh5ファイルからモデルを読み込んで学習する際のh5ファイルのパス
        :param monitor: モデルの途中で記録するパラメータ　デフォルトだと途中で記録しない
        :param save_weights_only:
        :param will_use_multi_inputs_per_one_image:
        :param data_preprocess:
        :return: 学習済みモデル
        """
        data_picker = BaggingDataPicker(pick_data_num, build_dataset_num)
        train_base_dir, validation_dir = self.build_train_validation_dir_paths(dataset_root_dir)
        bagging_dir = data_picker.copy_dataset_for_bagging(train_base_dir)
        model_base = [self.build_model(result_dir_path, result_name+index_name, tmp_model_path, monitor) for
                      index_name in os.listdir(bagging_dir)]
        bagging_train_dirs = [os.path.join(bagging_dir, dir_name) for dir_name in os.listdir(bagging_dir)]
        result_names = [result_name + dir_name for dir_name in os.listdir(bagging_dir)]
        print(result_names)
        return [self.train_with_validation_from_model(model,
                                                      result_dir_path,
                                                      bagging_train_dir,
                                                      validation_dir,
                                                      batch_size,
                                                      epoch_num,
                                                      result_model_name,
                                                      model_name,
                                                      save_weights_only,
                                                      will_use_multi_inputs_per_one_image,
                                                      data_preprocess)
                for model, bagging_train_dir, result_model_name in zip(model_base, bagging_train_dirs, result_names)]

    def build_validation_generator_and_get_steps_per_epoch(self,
                                                           train_dir: str,
                                                           validation_dir: str,
                                                           batch_size=32):
        train_generator = self.build_train_generator(batch_size, train_dir)
        test_generator = self.build_test_generator(batch_size, validation_dir)
        return train_generator, len(train_generator), test_generator, len(test_generator) - 1

