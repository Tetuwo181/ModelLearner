from torchvision.models.mobilenetv2 import mobilenet_v2
import torch
from torch.nn import Sequential, Dropout, Linear, ReLU, Softmax, Sigmoid
from util_types import types_of_loco


def builder(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        ) -> torch.nn.Module:
    base_model = mobilenet_v2(pretrained=True, width_mult=0.35)
    if class_num > 2:
        base_model.classifier = Sequential(
            Dropout(0.2),
            ReLU(base_model.last_channel),
            Linear(base_model.last_channel, class_num)
        )
    else:
        base_model.classifier = Sequential(
            Dropout(0.2),
            ReLU(base_model.last_channel),
            Linear(base_model.last_channel, 1),
            Sigmoid()
        )
    return base_model
