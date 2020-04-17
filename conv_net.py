import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ndf = 20 * 8 * 8 # 980
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, bias=True)
        self.fc1 = nn.Linear(self.ndf, 10, bias=False)
        # self.bn1 = nn.BatchNorm2d(10)
        # self.bn2 = nn.BatchNorm2d(20)
        self.loss = 0.0
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        # x = x.view(x.size(0), -1)
        x = x.view(-1, self.ndf)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return F.softmax(x)

    def get_loss(self):
        return self.loss

    def set_loss(self, loss):
        self.loss = loss

    def set_parameter(self, param_dict):
        st_dict = {}
        for key, value in param_dict.items():
            st_dict[key] = torch.nn.Parameter(torch.Tensor(value.float()))
        self.load_state_dict(st_dict)

# net = ConvNet()
