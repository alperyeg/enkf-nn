import torch
from enum import Enum
from torch import nn
from torch.nn import functional as F


class Views(nn.Module):
    def __init__(self, shape):
        super(Views, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.fc_shared = nn.Linear(100, 2)
        self.decision = self.decision_block()

    def forward(self, x1):
        x1 = x1.view(-1, 784)
        x1 = self.decision(x1)
        return x1

    def decision_block(self):
        return nn.Sequential(
            # nn.Conv2d(1, 20, 5),
            # nn.SELU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(20, 50, 5),
            # nn.SELU(),
            # Views((-1, self.ndf)),
            nn.Linear(784, 100),
            nn.SELU(),
            self.fc_shared,
            nn.SELU(),
            # nn.Linear(100, 2),
            # nn.SELU(),
            nn.Softmax(dim=1)
        )


class ClassificatorNet(nn.Module):
    def __init__(self):
        super(ClassificatorNet, self).__init__()
        self.ndf = 4 * 4 * 50
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LossFunctions(Enum):
    BCE = 'BCE'
    CE = 'CE'
    MAE = 'MAE'
    MSE = 'MSE'
    NORM = 'NORM'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


def loss_binary_decision(x, target, loss_name=LossFunctions.BCE):
    if loss_name == LossFunctions.BCE:
        if target == 0:
            targets = _shape_targets(x, target)
            loss = F.binary_cross_entropy(x, targets.to(torch.float),
                                          reduction='sum')
        else:
            targets = _shape_targets(x, target)
            loss = F.binary_cross_entropy(x, targets.to(torch.float),
                                          reduction='sum')
    elif loss_name == LossFunctions.MSE:
        if target == 0:
            loss = F.mse_loss(x, torch.zeros_like(x), reduction='sum')
        else:
            loss = F.mse_loss(x, torch.ones_like(x), reduction='sum')
    else:
        raise KeyError('Not known loss_name: {}'.format(loss_name))
    return loss


def _shape_targets(x, target):
    targets = torch.empty_like(x)
    if target == 0:
        targets[:, 0] = torch.zeros_like(x[:, 0])
        targets[:, 1] = torch.ones_like(x[:, 1])
    elif target == 1:
        targets[:, 0] = torch.ones_like(x[:, 0])
        targets[:, 1] = torch.zeros_like(x[:, 1])
    return targets
