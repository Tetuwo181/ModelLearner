import torch
from model_merger.pytorch.proc.loss.abstract_calculator import AbstractLossCalculator
from model_merger.pytorch.proc.loss.abstract_calculator import AbstractLossCalculatorForInceptionV3
from math import e as exponential


class AAEMCaluclator(object):

    def __init__(self, q):
        self.__q = q
    @property
    def q(self):
        return self.__q

    def l_plus(self, x):
        return (2/self.q)*torch.square(x)

    def l_minus(self, x: torch.Tensor):
        if x.is_cuda:
            use_device = x.get_device()
            pow_base = torch.tensor(self.q*exponential).to(use_device)
            return 2*torch.pow(pow_base, -((2.77/self.q)*x))
        pow_base = torch.tensor(self.q*exponential)
        return 2*torch.pow(pow_base, -((2.77/self.q)*x))


class AAEUMLoss(AAEMCaluclator, AbstractLossCalculator):
    """
    https://www.mdpi.com/2073-8994/10/9/385の論文に掲載されている損失関数
    """

    def __init__(self, q: float = 100):
        super(AAEUMLoss, self).__init__(q)

    def forward(self, distance, y):
        losses = y*self.l_plus(distance) + (1-y)*self.l_minus(distance)
        loss = torch.sum(losses) / (distance.size()[0])
        return loss


class AAEMLossForForInceptionV3(AAEMCaluclator, AbstractLossCalculatorForInceptionV3):

    def __init__(self, q):
        super(AAEMLossForForInceptionV3, self).__init__(q)

    def forward(self, distance, aux_distance, y, aux_y):
        losses = y*self.l_plus(distance) + (1-y)*self.l_minus(distance)
        loss = torch.sum(losses) / (distance.size()[0])
        aux_losses = aux_y*self.l_plus(aux_distance) + (1-aux_y)*self.l_minus(aux_distance)
        aux_loss = torch.sum(aux_losses) / (aux_distance.size()[0])
        return loss, aux_loss


class ContrastiveLoss(AbstractLossCalculator):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.__margin = margin

    def forward(self, distance, y):
        dist = torch.sqrt(distance)
        mdist = self.__margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * distance + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / y.size()[0]
        return loss
