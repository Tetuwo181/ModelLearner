import numpy as np


def build_batch_for_shame_learner(data_batch, teachers):
    data_teacher_pair = [(data, teacher) for data, teacher in zip(data_batch, teachers)]
    shuffled_dataset = np.random.permutation(data_teacher_pair)
    other_batch = np.array([data[0] for data in shuffled_dataset])
    other_teachers = [data[1] for data in shuffled_dataset]
    shame_labels = [build_shame_label(base_label, other_label) for (base_label, other_label)
                    in zip(teachers, other_teachers)]
    return [data_batch, other_batch], np.array(shame_labels, dtype="f4")


def build_shame_label(base_label, other_label):
    for base_index, other_index in zip(base_label, other_label):
        if base_index != other_index:
            return 0
    return 1
