import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
from torch import optim


# Activation functions
class Arctanh(nn.Module):
    """
    Inverse hyperbolic tangent
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.atan(x)


class Arcch(nn.Module):
    """
    Inverse hyperbolic cosecant aka area hyperbolic cosecant.
    A discontinous function.
    """

    def __init__(self):
        super().__init__()
        self.border_r = 2 * math.exp(1) / (math.exp(1) ** 2 - 1)
        self.border_l = -1 * self.border_r

    def forward(self, x):
        a = x.clone()
        arcch = torch.exp(1 / a + torch.sqrt(1 / a ** 2 + 1))
        ind_r = torch.where((a <= self.border_r) & (a >= 0))[0]
        ind_l = torch.where((a >= self.border_l) & (a <= 0))[0]
        arcch[ind_r] = 1
        arcch[ind_l] = -1
        return arcch


class Sinc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, normalize=False):
        x[x == 0] = 1
        if normalize:
            pi = torch.as_tensor(math.pi, device=device)
            return torch.sin(pi * x) / (pi * x)
        else:
            return torch.sin(x) / x


class ArSinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x + torch.sqrt(torch.pow(x, 2) + 1))


class Gauss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-torch.pow(x, 2))


class SQRBF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        x = a.clone()
        x[x <= 1] = torch.pow(x[x <= 1], 2) / 2
        x[(x >= 1) & (x < 2)] = (2 - x[(x >= 1) & (x < 2)]**2) / 2
        x[x > 2] = 0
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ndf = 20 * 8 * 8
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, bias=False)
        self.fc1 = nn.Linear(self.ndf, 10, bias=False)
        # self.bn1 = nn.BatchNorm2d(10)
        # self.bn2 = nn.BatchNorm2d(20)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.act_func = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.act_func(self.conv1(x)))
        x = self.act_func(self.conv2(x))
        # x = x.view(x.size(0), -1)
        x = x.view(-1, self.ndf)
        x = self.act_func(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return F.softmax(x, dim=1)

    def set_parameter(self, param_dict):
        st_dict = {}
        for key, value in param_dict.items():
            st_dict[key] = torch.nn.Parameter(
                torch.Tensor(value.cpu().float()))
        self.load_state_dict(st_dict)


def init_weights(m):
    mean = 0.
    std = 10.
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(1)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.fill_(1)

def get_data(batch_size, device):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    # Load data and normalize images to mean 0 and std 1
    # training set
    train_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='../', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    # test set
    test_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='../', train=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader_mnist, test_loader_mnist


def train(epoch, train_loader_mnist):
    net.train()
    train_loss = 0
    for idx, (img, target) in enumerate(train_loader_mnist):
        optimizer.zero_grad()
        # network prediction for the image
        output = net(img)
        # calculate the loss
        loss = criterion(output, target)
        # backprop
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if idx % 10 == 0:
            print('Loss {} in epoch {}, idx {}'.format(
                loss.item(), epoch, idx))

    print('Average loss: {} epoch:{}'.format(
        train_loss / len(train_loader_mnist.dataset), epoch))


def test(epoch, test_loader_mnist):
    net.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            output = net(img)
            loss = criterion(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

            if idx % 10 == 0:
                print('Test Loss {} in epoch {}, idx {}'.format(
                    loss.item(), epoch, idx))

        print('Test accuracy: {} Average test loss: {} epoch:{}'.format(
            100 * test_accuracy / len(test_loader_mnist.dataset),
            test_loss / len(test_loader_mnist.dataset), epoch))


if __name__ == '__main__':
    net = ConvNet()
    net.apply(init_weights)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # Cross entropy loss to calculate the loss
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 64
    train_loader, test_loader = get_data(batch, device)
    for ep in range(1, 2):
        train(ep, train_loader)
        print('training done')
        test(ep, test_loader)
