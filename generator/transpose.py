import numpy as np


def transpose(batch):
    return np.array([np.transpose(params) for params in batch])


def transpose_preprocess(x, y):
    return transpose(x), y
