import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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
        x[(x >= 1) & (x < 2)] = (2 - x[(x >= 1) & (x < 2)] ** 2) / 2
        x[x > 2] = 0
        return x


class HardShrinkMod(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lam=0.5, alpha=1):
        y = torch.zeros_like(x)
        y[x > alpha] = alpha
        ind = (lam < x) & (x < alpha)
        y[ind] = x[ind]
        ind = (-alpha < x) & (x < -lam)
        y[ind] = x[ind]
        y[x < -alpha] = -alpha
        return y


class SigmoidMod(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-0.4 * x))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.ndf = 20 * 8 * 8
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, bias=False)
        self.fc1 = nn.Linear(self.ndf, 10, bias=False)

        self.act_func = nn.Sigmoid()

    def forward(self, x):
        act_func1 = self.act_func(self.conv1(x))
        x = self.pool(act_func1)
        x = self.act_func(self.conv2(x))
        act_func2 = x.clone()
        # x = x.view(x.size(0), -1)
        x = x.view(-1, self.ndf)
        x = self.fc1(x)
        return x, act_func1, act_func2

    def set_parameter(self, param_dict):
        st_dict = {}
        for key, value in param_dict.items():
            st_dict[key] = torch.nn.Parameter(
                torch.Tensor(value.cpu().float()))
        self.load_state_dict(st_dict)


def init_weights(m):
    mean = 0.
    # std = 5.
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.fill_(1)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.fill_(1)


def get_data(batch_size, device):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    # Load data and normalize images to mean 0 and std 1
    # training set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.], [1.])])
    train_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='../', train=True, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    # test set
    test_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='../', train=False,
                       transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader_mnist, test_loader_mnist


def train(epoch, train_loader_mnist, optimizer):
    net.train()
    train_loss = 0
    act_func = {'act1': [], 'act2': [], 'act1_mean': [], 'act2_mean': [],
                'act1_std': [], 'act2_std': [], 'act3': [], 'act3_mean': [],
                'act3_std': []}
    grads = {'conv1_grad': [], 'conv2_grad': [], 'fc1_grad': [],
             'conv1_grad_mean': [], 'conv2_grad_mean': [], 'fc1_grad_mean': [],
             'conv1_grad_std': [], 'conv2_grad_std': [], 'fc1_grad_std': []}
    act_mean_std = []
    for idx, (img, target) in enumerate(train_loader_mnist):
        optimizer.zero_grad()
        # network prediction for the image
        output, act1, act2 = net(img)
        act3 = F.softmax(output, dim=1)
        act_func['act1_mean'].append(act1.mean().item())
        act_func['act2_mean'].append(act2.mean().item())
        act_func['act3_mean'].append(act3.mean().item())
        act_func['act1_std'].append(act1.std().item())
        act_func['act2_std'].append(act2.std().item())
        act_func['act3_std'].append(act3.std().item())
        # calculate the loss
        loss = criterion(output, target)
        # backprop
        loss.backward()
        grads['conv1_grad_mean'].append(net.conv1.weight.grad.mean().item())
        grads['conv2_grad_mean'].append(net.conv2.weight.grad.mean().item())
        grads['fc1_grad_mean'].append(net.fc1.weight.grad.mean().item())
        grads['conv1_grad_std'].append(net.conv1.weight.grad.std().item())
        grads['conv2_grad_std'].append(net.conv2.weight.grad.std().item())
        grads['fc1_grad_std'].append(net.fc1.weight.grad.std().item())

        train_loss += loss.item()
        optimizer.step()

        if idx % 8 == 0:
            print('Loss {} in epoch {}, idx {}'.format(
                loss.item(), epoch, idx))
            grads['conv1_grad'].append(net.conv1.weight.grad.detach().numpy())
            grads['conv2_grad'].append(net.conv2.weight.grad.detach().numpy())
            grads['fc1_grad'].append(net.fc1.weight.grad.detach().numpy())
            act_func['act1'].append(act1.detach().numpy())
            act_func['act2'].append(act2.detach().numpy())
            act_func['act3'].append(act3.detach().numpy())
            # torch.save(net.state_dict(), 'results/model_it{}.pt'.format(idx))

    print('Average loss: {} epoch:{}'.format(
        train_loss / len(train_loader_mnist.dataset), epoch))
    if epoch % 1 == 0:
        # np.save('{}_gradients_ep{}.npy'.format(optimizer.__class__.__name__, epoch), grads)
        # np.save('{}_act_func_ep{}.npy'.format(optimizer.__class__.__name__, epoch), act_func)
        pass


def test(epoch, test_loader_mnist, optimizer):
    net.eval()
    test_accuracy = 0
    test_loss = 0
    ta = []
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            output, _, _ = net(img)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

            if idx % 8 == 0:
                tmp_acc = pred.eq(target.view_as(pred)).sum().item() * 100 / len(target)
                ta.append(tmp_acc)
                print('Test Loss {} in epoch {}, idx {}'.format(
                    loss.item(), epoch, idx))
                print('Test accuracy {} in epoch {}, idx {}'.format(tmp_acc, epoch, idx))

    print('Test accuracy: {} Average test loss: {} epoch:{}'.format(
        100 * test_accuracy / len(test_loader_mnist.dataset),
        test_loss / len(test_loader_mnist.dataset), epoch))
    # if 100 * test_accuracy / len(test_loader_mnist.dataset) >= 80:
    #     with open('std_optim_out.txt', 'a') as f:
    #         opt_name = optimizer.__class__.__name__
    #         ta = 100 * test_accuracy / len(test_loader_mnist.dataset)
    #         print(std, opt_name, ta, epoch, file=f)
    #         return True
    # else:
    #     return False
    torch.save(ta, '{}_test_accuracy_iteration{}'.format(optimizer.__class__.__name__, 
                                                         epoch))
    return 100 * test_accuracy / len(test_loader_mnist.dataset)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    net = ConvNet()
    # net.apply(init_weights)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.RMSprop(net.parameters())
    # optimizer = optim.Adagrad(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=0.9)
    # Cross entropy loss to calculate the loss
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Runs on {}'.format(device))
    batch = 64
    train_loader, test_loader = get_data(batch, device)
    # for ep in range(0, 50):
    #     train(ep, train_loader)
    #     print('training done')
    #     test(ep, test_loader)
    adam = optim.Adam(net.parameters(), lr=1e-3)
    sgd = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    rmsprop = optim.RMSprop(net.parameters())
    adagrad = optim.Adagrad(net.parameters())
    # optimizers = [sgd, adam, rmsprop, adagrad]
    optimizers = [adam, sgd]
    # stds = [0.01, 0.1, 1, 3, 5, 10]
    stds = [1]
    accs = []
    # with open('std_optim_out.txt', 'w') as f:
    #    print('std, optimizer, test_accuracy, epoch', file=f)
    for opt in optimizers:
        for s in stds:
            std = s
            net.apply(init_weights)
            acc_reached = False
            for ep in range(1, 11):
                train(ep, train_loader, opt)
                print('training done')
                acc_reached = test(ep, test_loader, opt)
                accs.append((s, acc_reached, opt.__class__.__name__))
                # if acc_reached:
                #     break
    # torch.save(accs, 'test_acc.pt')
    print(accs)
