import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(784, 60, bias=True)
        self.fc2 = nn.Linear(60, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def set_parameter(self, param_dict):
        st_dict = {}
        for key, value in param_dict.items():
            st_dict[key] = torch.nn.Parameter(torch.Tensor(value))
        self.load_state_dict(st_dict)
