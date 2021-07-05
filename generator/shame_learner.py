import numpy as np


def build_batch_for_shame_learner(data_batch, teachers):
    data_teacher_pair = [(data, teacher) for data, teacher in zip(data_batch, teachers)]
    shuffled_dataset = np.random.permutation(data_teacher_pair)
    other_batch = [data[0] for data in shuffled_dataset]
    other_teachers = [data[1] for data in shuffled_dataset]
    return data_batch, other_batch, teachers, other_teachers
