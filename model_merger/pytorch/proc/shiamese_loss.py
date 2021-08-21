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
        self.calc_loss(x0, x1, y)

    def calc_distance(self, x0, x1):
        return self.__distance_calculator(x0, x1)

    def calc_loss(self, x0, x1, y):
        distance = self.__distance_calculator.forward(x0, x1)
        loss = self.__loss_calculator.forward(distance, y)
        return loss


class SiameseLossForInceptionV3(SiameseLoss):

    def __init__(self,
                 distance_calculator: AbstractDistanceCaluclator = L1Norm(),
                 loss_calculator: AbstractLossCalculator = AAEUMLoss()):
        super(SiameseLossForInceptionV3, self).__init__(distance_calculator, loss_calculator)
        
    def forward(self, params, y):
        x0 = params[0]
        x1 = params[1]
        print(x0, x1)
        main_y = y[0]
        aux_y = y[1]
        main_loss = self.calc_loss(x0.logits, x1.logits, main_y)
        aux_loss = self.calc_loss(x0.aux_logits, x1.aux_logits, aux_y)
        return main_loss, aux_loss

