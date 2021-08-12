import numpy as np
from generator.transpose import transpose


def build_batch_for_siameselearner(data_batch, teachers):
    data_teacher_pair = [(data, teacher) for data, teacher in zip(data_batch, teachers)]
    shuffled_dataset = np.random.permutation(data_teacher_pair)
    other_batch = np.array([data[0] for data in shuffled_dataset])
    other_teachers = [data[1] for data in shuffled_dataset]
    shame_labels = [build_shame_label(base_label, other_label) for (base_label, other_label)
                    in zip(teachers, other_teachers)]
    return [data_batch, other_batch], np.array(shame_labels, dtype="f4")


def build_shame_label(base_label, other_label):
    if type(base_label) is np.float32:
        return 1 if int(base_label) == int(other_label) else 0
    for base_index, other_index in zip(base_label, other_label):
        if base_index != other_index:
            return 0
    return 1


def build_batchbuilder_for_siamese(will_transpose: bool, convert_numpy: bool = False):

    def transpose_builder(data_batch, teachers):
        use_batch = transpose(data_batch)
        return build_batch_for_siameselearner(use_batch, teachers)

    def transpose_builder_with_convert_numpy(data_batch, teachers):
        batch, teachers = transpose_builder(data_batch, teachers)
        return np.array(batch), teachers

    def build_batch_for_siameselearner_with_convert_numpy(data_batch, teachers):
        batch, teachers = build_batch_for_siameselearner(data_batch, teachers)
        return np.array(batch), teachers

    if convert_numpy:
        return transpose_builder_with_convert_numpy if will_transpose else build_batch_for_siameselearner_with_convert_numpy
    return transpose_builder if will_transpose else build_batch_for_siameselearner

