from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot
from tensorflow.python.framework.ops import Tensor
from typing import Union, Callable

Merge = Union[Add, Subtract, Multiply, Average, Minimum, Maximum, Concatenate, Dot]
Loss = Union[str, Callable[[Tensor, Tensor], Tensor]]
TrainableModelIndex = Union[bool, int]
