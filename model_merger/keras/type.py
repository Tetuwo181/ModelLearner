from __future__ import annotations
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot
from tensorflow import Tensor
from typing import Union, Callable

Merge = Add | Subtract | Multiply | Average | Minimum | Maximum | Concatenate | Dot
Loss = str | Callable[[Tensor, Tensor], Tensor]
TrainableModelIndex = bool | int
