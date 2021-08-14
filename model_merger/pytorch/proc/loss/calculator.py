import torch
from model_merger.pytorch.proc.loss.abstract_calculator import AbstractLossCalculator
from math import e as exponential


class AAEUMLoss(AbstractLossCalculator):
    """
    https://www.mdpi.com/2073-8994/10/9/385の論文に掲載されている損失関数
    """

    def __init__(self, q: float = 100):
        super(AAEUMLoss, self).__init__()
        self.__q = q

    @property
    def q(self):
        return self.__q

    def l_plus(self, x):
        return (2/self.q)*torch.square(x)

    def l_minus(self, x: torch.Tensor):
        return 2*self.q*torch.exp(-((2.77/self.q)*x))

    def forward(self, distance, y):
        losses = y*self.l_plus(distance) + (1-y)*self.l_minus(distance)
        loss = torch.sum(losses) / (distance.size()[0])
        return loss
