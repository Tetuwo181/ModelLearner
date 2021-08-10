import torch
from model_merger.pytorch.proc.distance.abs_calculator import AbstractDistanceCaluclator
from model_merger.pytorch.proc.distance.calculator import L1Norm
from model_merger.pytorch.proc.loss.abstract_calculator import AbstractLossCalculator
from model_merger.pytorch.proc.loss.calculator import AAEUMLoss


class SiameseLoss(torch.nn.Module):

    def __init__(self,
                 distance_calculator: AbstractDistanceCaluclator = L1Norm(),
                 loss_calculator: AbstractLossCalculator = AAEUMLoss()):
        super(SiameseLoss, self).__init__()

        self.__distance_calculator = distance_calculator
        self.__loss_calculator = loss_calculator

    def forward(self, params, y):

        x0 = params[0]
        x1 = params[1]
        distance = self.__distance_calculator.forward(x0, x1)
        loss = self.__loss_calculator.forward(distance, y)
        return loss

    def calc_distance(self, x0, x1):
        return self.__distance_calculator(x0, x1)
