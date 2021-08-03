from torchvision.models.mobilenetv2 import mobilenet_v2
import torch
from torch.nn import Sequential, Dropout, Linear
from util_types import types_of_loco
from torchinfo import summary


def builder(
        class_num: int,
        img_size: types_of_loco.input_img_size = 28,
        channels: int = 3,
        ) -> torch.nn.Module:
    base_model = mobilenet_v2(pretrained=True)
    base_model.classifier = Sequential(
        Dropout(0.2),
        Linear(base_model.last_channel, class_num),
    )
    summary(base_model, input_size=(32, 3, 224, 224), col_names=["output_size", "num_params"])
    return base_model
