from conv_net import ConvNet
import numpy as np
import torch
from torchvision import datasets, transforms


def shape_parameter_to_conv_net(net, params):
    param_dict = dict()
    start = 0
    for key in net.state_dict().keys():
        shape = net.state_dict()[key].shape
        length = net.state_dict()[key].nelement()
        end = start + length
        param_dict[key] = params[start:end].reshape(shape)
        start = end
    return param_dict


def get_data(batch_size, device):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    # Load data and normalize images to mean 0 and std 1
    # training set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.1307], [0.3081])
         ])
    train_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='../../../../../', train=True, download=False,
                       transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)
    # test set
    test_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(root='../../../../../', train=False, download=False,
                       transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader_mnist, test_loader_mnist


def score(x, y):
    """
    :param x: targets
    :param y: prediction
    :return:
    """
    # print('target ', x)
    # print('predict ', y)
    n_correct = np.count_nonzero(y == x)
    n_total = len(y)
    sc = n_correct / n_total
    return sc


def test(net, test_loader_mnist):
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            output, _, _ = net(img)
            loss = criterion(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            ta = pred.eq(target.view_as(pred)).sum().item()
            test_accuracy += ta
            test_acc = score(target.cpu().numpy(),
                             np.argmax(output.cpu().numpy(), 1))
            if idx % 10 == 0:
                print('Test Loss {}, idx {}'.format(
                    loss.item(), idx))

        print('Test accuracy: {} Average test loss: {}'.format(
            100 * test_accuracy / len(test_loader_mnist.dataset),
            test_loss / len(test_loader_mnist.dataset)))


if __name__ == '__main__':
    conv_params = torch.load('conv_params.pt', map_location='cpu')
    ensemble = conv_params['ensemble'].mean(0)
    ensemble = torch.from_numpy(ensemble)
    conv_net = ConvNet()
    ds = shape_parameter_to_conv_net(conv_net, ensemble)
    conv_net.set_parameter(ds)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 64
    train_loader, test_loader = get_data(batch, device)
    test(conv_net, test_loader)
