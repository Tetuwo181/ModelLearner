from torchvision.models.mobilenetv2 import mobilenet_v2
import torch
from util_types import types_of_loco


def builder(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        ) -> torch.nn.Module:
    return mobilenet_v2(pretrained=True, num_classes=class_num)
