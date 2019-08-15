import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ndf = 20 * 8 * 8
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, bias=False)
        self.fc1 = nn.Linear(self.ndf, 10, bias=False)
        self.fc1_bin = nn.Linear(self.ndf, 2, bias=False)
        self.loss = 0.0

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.ndf)
        y = F.relu(self.fc1_bin(x))
        x = F.relu(self.fc1(x))
        return F.softmax(x, dim=1), F.softmax(y, dim=1)

    def get_loss(self):
        return self.loss

    def set_loss(self, loss):
        self.loss = loss

    def set_parameter(self, param_dict):
        st_dict = {}
        for key, value in param_dict.items():
            st_dict[key] = torch.nn.Parameter(torch.Tensor(value))
        self.load_state_dict(st_dict)

# net = ConvNet()
