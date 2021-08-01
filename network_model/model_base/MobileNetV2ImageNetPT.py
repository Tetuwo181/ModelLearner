from torchvision.models.mobilenetv2 import mobilenet_v2
import torch
from torch import nn
from util_types import types_of_loco


def builder(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        ) -> torch.nn.Module:
    base_model = mobilenet_v2(pretrained=True)
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(base_model.last_channel, class_num),
    )
    print(base_model.parameters())
    return base_model
