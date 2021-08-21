import numpy as np
from generator.transpose import transpose
from typing import Optional


def build_batch_for_siameselearner(data_batch, teachers, margin=1, build_set_num=1):
    return build_other_teacher_and_labels(data_batch, teachers, margin, build_set_num)


def build_other_batch(data_batch, teachers):
    data_teacher_pair = [(data, teacher) for data, teacher in zip(data_batch, teachers)]
    shuffled_dataset = np.random.permutation(data_teacher_pair)
    other_batch = np.array([data[0] for data in shuffled_dataset])
    other_teachers = [data[1] for data in shuffled_dataset]
    return other_batch, other_teachers


def build_siamese_labels(teachers, other_teachers, margin):
    return [build_siamese_label(base_label, other_label, margin) for (base_label, other_label)
            in zip(teachers, other_teachers)]


def build_siamese_labels_for_space(teachers, other_teachers, margin):
    return [build_siamese_label_for_space(base_label, other_label, margin) for (base_label, other_label)
            in zip(teachers, other_teachers)]


def build_other_teacher_and_label(data_batch, teachers, margin=1):
    other_batch, other_teachers = build_other_batch(data_batch, teachers)
    shame_labels = build_siamese_labels(data_batch, teachers, margin)
    return data_batch, other_batch, shame_labels


def build_other_teacher_and_labels(data_batch, teachers, margin=1, build_set_num=1):
    use_data_batch = []
    use_other_batch = []
    use_labels = []
    for _ in range(build_set_num):
        built_batch, built_other, built_labels = build_other_teacher_and_label(data_batch, teachers, margin)
        for data in built_batch:
            use_data_batch.append(data)
        for data in built_other:
            use_other_batch.append(data)
        for label in built_labels:
            use_labels.append(label)
    return [np.array(use_data_batch), np.array(use_other_batch)], np.array(use_labels, dtype="f4")


def build_siamese_label_for_space(base_label, other_label, margin=1):
    return 1 if np.abs(base_label - other_label) < margin else 0


def build_siamese_label(base_label, other_label, margin=1):
    if type(base_label) is np.float32:
        return build_siamese_label_for_space(base_label, other_label, margin)
    for base_index, other_index in zip(base_label, other_label):
        if base_index != other_index:
            return 0
    return 1


def build_batchbuilder_for_siamese(will_transpose: bool,
                                   convert_numpy: bool = False,
                                   margin=1,
                                   build_set_num=1,
                                   aux_margin=1):

    def transpose_builder(data_batch, teachers, will_use_aux: bool = False):
        use_batch = transpose(data_batch)
        use_margin = aux_margin if will_use_aux else margin
        return build_batch_for_siameselearner(use_batch, teachers, use_margin, build_set_num)

    def transpose_builder_with_convert_numpy(data_batch, teachers, will_use_aux: bool = False):
        batch, teachers = transpose_builder(data_batch, teachers)
        return np.array(batch), teachers

    def build_batch_for_siameselearner_with_convert_numpy(data_batch, teachers, will_use_aux: bool = False):
        use_margin = aux_margin if will_use_aux else margin
        batch, teachers = build_batch_for_siameselearner(data_batch, teachers, use_margin, build_set_num)
        return np.array(batch), teachers

    if convert_numpy:
        return transpose_builder_with_convert_numpy if will_transpose else build_batch_for_siameselearner_with_convert_numpy
    return transpose_builder if will_transpose else build_batch_for_siameselearner


class SiameseLearnerDataBuilder(object):

    def __init__(self,
                 will_transpose: bool,
                 convert_numpy: bool,
                 build_set_num: int,
                 margin: int,
                 aux_margin: Optional[int] = None):

        self.__will_transpose = will_transpose
        self.__build_set_num = build_set_num
        self.__margin = margin
        self.__aux_margin = margin if aux_margin is None else aux_margin
        self.__data_builder = build_batchbuilder_for_siamese(will_transpose,
                                                             convert_numpy,
                                                             margin,
                                                             build_set_num,
                                                             self.__aux_margin)

    @property
    def margin(self):
        return self.__margin

    @property
    def aux_margin(self):
        return self.__aux_margin

    @property
    def will_transpose(self):
        return self.__will_transpose

    @property
    def build_set_num(self):
        return self.__build_set_num

    def __call__(self, data_batch, teachers, will_use_aux: bool = False):
        return self.__data_builder(data_batch, teachers, will_use_aux)
