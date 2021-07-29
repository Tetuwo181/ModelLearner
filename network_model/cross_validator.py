from network_model.wrapper.keras import many_data as md
from sklearn.model_selection import StratifiedKFold
import keras.engine.training
from keras.preprocessing.image import ImageDataGenerator
from typing import Callable
from typing import List
from typing import Optional
import numpy as np
import copy
import gc
from DataIO import data_loader as dl
from util_types import two_dim
from network_model.builder import init_loader_setting


img_size, size_converter = two_dim.init_pair_type(int)


def build_input_image(model_builder: Callable[[int], keras.engine.training.Model],
                      class_set: List[str],
                      result_dir_path: str,
                      fold_num: int,
                      epoch_num: int,
                      image_generator: ImageDataGenerator = None,
                      generator_batch_size: int = 32
                      ):
    """
    交差検証を行うための関数を生成する
    :param model_builder: モデル生成器
    :param class_set: クラスの元となったリスト
    :param result_dir_path: 結果を記録する際のパス
    :param fold_num: 交差検証を行う回数
    :param epoch_num: エポック数
    :param image_generator: keras形式でのデータを水増しするジェネレータ これを引数で渡さない場合はデータの水増しをしない
    :param generator_batch_size: ジェネレータのバッチサイズ
    :return:
    """
    skf = StratifiedKFold(n_splits=fold_num)

    def test(data_set: np.array, label_set: np.ndarray, result_name: str = "result", model_name: str = "model"):
        """
        実際にバリデーションを行う
        :param data_set: データセット
        :param label_set: データのラベル
        :param result_name: 結果の出力先名
        :param model_name: モデルの名前
        :return:
        """
        label_index_set = [np.argmax(label) for label in label_set]
        for fold_itr, (train_index, test_index) in enumerate(skf.split(data_set, label_index_set)):
            print("iteration", fold_itr, "start")
            copied_generator = copy.deepcopy(image_generator)
            train_data = np.array([data_set[index] for index in train_index])
            train_label = np.array([label_set[index] for index in train_index])
            test_data = np.array([data_set[index] for index in test_index])
            test_label = np.array([label_set[index] for index in test_index])
            model_name_iter = model_name + str(fold_itr)
            model = md.Model(model_builder(len(class_set)), class_set)
            model.test(train_data,
                       train_label,
                       test_data,
                       test_label,
                       epoch_num,
                       image_generator=copied_generator,
                       generator_batch_size=generator_batch_size,
                       result_dir_name=result_name,
                       dir_path=result_dir_path,
                       model_name=model_name_iter,
                       will_del_from_ram=True)
            # メモリ対策のため削除（モデルの記録自体保存しているため、わざわざメモリ上に残す必要もない）
            model = None
            del model
            gc.collect()
            copied_generator = None
            train_data = None
            test_data = None
            train_label = None
            test_label = None
            del train_data
            del test_data
            del train_label
            del test_label
            del copied_generator
            gc.collect()
            print(model_name_iter, "deleted")
        model = md.Model(model_builder(len(class_set)), class_set)
        model.fit(data_set, label_set, epoch_num).record(result_name, result_dir_path, model_name)
    return test


def build_input_type_dir(model_builder: Callable[[int], keras.engine.training.Model],
                         dataset_root_dir: str,
                         result_dir_path: str,
                         fold_num: int = 5,
                         epoch_num: int = 20,
                         image_generator: ImageDataGenerator = None,
                         generator_batch_size: int = 32,
                         normalize_type: dl.NormalizeType = dl.NormalizeType.Div255,
                         img_resize_val: Optional[img_size] = None,
                         color: str = "RGB"
                         ):
    """
    交差検証を行うための関数を生成する
    :param model_builder: モデル生成器
    :param dataset_root_dir: データの格納されているディレクトリ
    :param result_dir_path: 結果を記録する際のパス
    :param fold_num: 交差検証を行う回数
    :param epoch_num: エポック数
    :param image_generator: keras形式でのデータを水増しするジェネレータ これを引数で渡さない場合はデータの水増しをしない
    :param generator_batch_size: ジェネレータのバッチサイズ
    :param normalize_type: どのように正規化するか
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :param color: グレースケールかカラーで読み込むか　デフォルトではカラー(RGB)
    :return:
    """

    dataset_paths, label_set, class_names, class_num = dl.load_dataset_path(dataset_root_dir)
    print("data_num:", len(dataset_paths))
    skf = StratifiedKFold(n_splits=fold_num)

    def test_load_from_path(
                            result_name: str = "result",
                            model_name: str = "model"
                            ):
        """
        実際にバリデーションを行う
        データ読み込みをこの中で行う
        :param result_name: 結果の出力先名
        :param model_name: モデルの名前
        :return:
        """
        label_index_set = [np.argmax(label) for label in label_set]
        for fold_itr, (train_index, test_index) in enumerate(skf.split(dataset_paths, label_index_set)):
            print("iteration", fold_itr, "start")
            copied_generator = copy.deepcopy(image_generator)
            # 教師データ読み込み
            train_data_pathes = [dataset_paths[index] for index in train_index]
            print("load teacher data. data num;", len(train_data_pathes))
            train_data = np.array([dl.load_img(img_path, img_resize_val, color) for img_path in train_data_pathes])
            train_data = dl.normalise_img_set(train_data, normalize_type)
            train_label = np.array([label_set[index] for index in train_index])

            # テストデータ読み込み
            test_data_pathes = [dataset_paths[index] for index in test_index]
            print("load test data data num;", len(test_data_pathes))
            test_data = np.array([dl.load_img(img_path, img_resize_val, color) for img_path in test_data_pathes])
            test_data = dl.normalise_img_set(test_data, normalize_type)
            test_label = np.array([label_set[index] for index in test_index])

            # モデル名など設定
            model_name_iter = model_name + str(fold_itr)
            model = md.Model(model_builder(len(class_names)), class_names)

            print("data is loaded")

            # テスト開始
            model.test(train_data,
                       train_label,
                       test_data,
                       test_label,
                       epoch_num,
                       normalize_type=normalize_type,
                       image_generator=copied_generator,
                       generator_batch_size=generator_batch_size,
                       result_dir_name=result_name,
                       dir_path=result_dir_path,
                       model_name=model_name_iter,
                       will_del_from_ram=True)
            # メモリ対策のため削除（モデルの記録自体保存しているため、わざわざメモリ上に残す必要もない）
            model = None
            del model
            gc.collect()
            copied_generator = None
            train_data = None
            test_data = None
            train_label = None
            test_label = None
            del train_data
            del test_data
            del train_label
            del test_label
            del copied_generator
            gc.collect()
            print(model_name_iter, "deleted")
        model = md.Model(model_builder(len(class_names)), class_names)
        data_set = np.array([dl.load_img(img_path, img_resize_val, color) for img_path in dataset_paths])
        model.fit(data_set, label_set, epoch_num)\
             .record(result_name,
                     result_dir_path,
                     model_name,
                     normalize_type=normalize_type
                     )

    return test_load_from_path


def build_input_type_dir_for_large_dataset(model_builder: Callable[[int], keras.engine.training.Model],
                                           dataset_root_dir: str,
                                           result_dir_path: str,
                                           fold_num: int = 5,
                                           epoch_num: int = 20,
                                           image_generator: ImageDataGenerator = None,
                                           generator_batch_size: int = 32,
                                           build_original_data_num: int = 4,
                                           test_generator_batch_size: int = 32,
                                           normalize_type: dl.NormalizeType = dl.NormalizeType.Div255,
                                           img_resize_val: Optional[img_size] = None,
                                           color: str = "RGB"
                                           ):
    """
    交差検証を行うための関数を生成する
    データがメモリに乗りきらない場合はこちらを使う
    :param model_builder: モデル生成器
    :param dataset_root_dir: データの格納されているディレクトリ
    :param result_dir_path: 結果を記録する際のパス
    :param fold_num: 交差検証を行う回数
    :param epoch_num: エポック数
    :param image_generator: keras形式でのデータを水増しするジェネレータ これを引数で渡さない場合はデータの水増しをしない
    :param generator_batch_size: ジェネレータのバッチサイズ　データを水増ししない場合はこれがこのままのサイズで渡され,水増しする場合はbuild_original_data_num倍した分だけデータが水増しされる
    :param build_original_data_num: 1回の試行あたりでまとめて水増しするデータの数
    :param test_generator_batch_size: 検証用データのバッチサイズ
    :param normalize_type: どのように正規化するか
    :param generator_batch_size: ジェネレータのバッチサイズ　データを水増ししない場合はこれがこのままのサイズで渡され,水増しする場合はbuild_original_data_num倍した分だけデータが水増しされる
    :param img_resize_val: 画像のサイズをリサイズする際のサイズ　指定しなければオリジナルのサイズのまま読み込み
    :param color: グレースケールかカラーで読み込むか　デフォルトではカラー(RGB)
    :return:
    """

    dataset_paths, label_set, class_names, class_num = dl.load_dataset_path(dataset_root_dir)
    print("data_num:", len(dataset_paths))
    data_loader_base = init_loader_setting(class_num, img_resize_val, color, normalize_type)
    train_loader_base = data_loader_base(image_generator)

    def test_load_from_path(
                            result_name: str = "result",
                            model_name: str = "model"
                            ):
        """
        実際にバリデーションを行う
        データ読み込みをこの中で行う
        :param result_name: 結果の出力先名
        :param model_name: モデルの名前
        :return:
        """
        skf = StratifiedKFold(n_splits=fold_num)
        label_index_set = [np.argmax(label) for label in label_set]
        for fold_itr, (train_index, test_index) in enumerate(skf.split(dataset_paths, label_index_set)):
            print("iteration", fold_itr, "start")
            # 教師データ読み込み
            train_data_paths = [dataset_paths[index] for index in train_index]
            print("load teacher data. data num;", len(train_data_paths))
            train_label = [label_set[index] for index in train_index]
            train_generator = train_loader_base(train_data_paths, train_label, generator_batch_size) \
                if image_generator is None else train_loader_base(train_data_paths,
                                                                  train_label,
                                                                  generator_batch_size,
                                                                  build_original_data_num)

            # テストデータ読み込み
            test_data_paths = [dataset_paths[index] for index in test_index]
            print("load test data data num;", len(test_data_paths))
            test_label = [label_set[index] for index in test_index]
            test_generator = data_loader_base()(test_data_paths, test_label, test_generator_batch_size)

            # モデル名など設定
            model_name_iter = model_name + str(fold_itr)
            model = md.ModelForManyData(model_builder(len(class_names)), class_names)

            print("data is loaded")

            # テスト開始
            model.test(train_generator,
                       epoch_num,
                       test_generator,
                       normalize_type=normalize_type,
                       result_dir_name=result_name,
                       dir_path=result_dir_path,
                       model_name=model_name_iter,
                       will_del_from_ram=True)
        learn_all_data(result_dir_path, model_name)

    def learn_all_data(result_name: str = "result",
                       model_name: str = "model"
                       ):
        """
        バリデーションを行わず学習のみ行う
        :param result_name: 結果の出力先名
        :param model_name: モデルの名前
        :return:
        """
        model = md.ModelForManyData(model_builder(len(class_names)), class_names)
        train_generator = train_loader_base(dataset_paths, label_set, generator_batch_size) \
            if image_generator is None else train_loader_base(dataset_paths,
                                                              label_set,
                                                              generator_batch_size,
                                                              build_original_data_num)
        model.fit_generator(train_generator, epoch_num)\
             .record(result_name,
                     result_dir_path,
                     model_name,
                     normalize_type=normalize_type
                     )

    return test_load_from_path if fold_num > 1 else learn_all_data

