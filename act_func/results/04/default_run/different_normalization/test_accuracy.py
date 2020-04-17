from conv_net import ConvNet
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from torchvision import datasets, transforms

sns.set(style="white")
sns.set_color_codes("dark")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.})


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
         transforms.Normalize([0], [1])
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
                print('Test Loss {}, idx {}'.format(loss.item(), idx))
        ta = 100 * test_accuracy / len(test_loader_mnist.dataset)
        tl = test_loss / len(test_loader_mnist.dataset)
        print('Test accuracy: {} Average test loss: {}'.format(ta, tl))
        return ta, tl


def plot_loss(losses, iters, title='Cost'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$c$')
    ax.plot(range(len(losses)), losses, '.-')
    plt.xticks(range(len(iters)), iters)
    plt.tight_layout()
    plt.show()
    # fig.savefig(title + '.pdf')


def plot_accuracy(accuracies, iters):
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(accuracies, '.-')
    ax1.set_ylabel('Accuracy in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    # ax1.set_title()
    plt.tight_layout()
    plt.show()
    fig.savefig('test_accuracy.pdf')


def plot_different_accuracies(iters):
    norm_loss = 100 - np.array(torch.load('acc_loss.pt')[0])
    more_loss = 100 - np.array(torch.load('more_ensembles_acc_loss.pt')[0])
    less_loss = 100 - np.array(torch.load('less_ensembles_acc_loss.pt')[0])
    fig, ax1 = plt.subplots()
    ax1.plot(less_loss, '.-', label='100 ensembles')
    ax1.plot(norm_loss, '.-', label='5000 ensembles')
    ax1.plot(more_loss, '.-', label='10000 ensembles')
    ax1.set_ylabel('Test Error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    # ax1.set_title()
    # plt.tight_layout()
    plt.legend()
    plt.show()
    fig.savefig('ensembles_diff_test_accuracies.pdf')


def plot_act_func_accuracies(iters):
    norm_loss = 100 - np.array(torch.load('acc_loss.pt')[0])
    more_loss = 100 - np.array(torch.load('relu_acc_loss.pt')[0])
    less_loss = 100 - np.array(torch.load('tanh_acc_loss.pt')[0])
    print(norm_loss)
    print(more_loss)
    print(less_loss)
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(norm_loss, '.-', label='Logistic Function')
    ax1.plot(more_loss, '.-', label='ReLU')
    ax1.plot(less_loss, '.-', label='Tanh')
    ax1.set_ylabel('Test Error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    # ax1.set_title()
    # plt.tight_layout()
    plt.legend()
    plt.show()
    fig.savefig('act_func_test_accuracies.pdf', bbox_inches='tight',
                pad_inches=0.1)


def plot_test_error(accuracies, iters):
    fig, ax1 = plt.subplots()
    acc = 100 - np.array(accuracies)
    ax1.plot(acc, '.-', markersize=6., label=r'EnKF $\sigma=1$')
    ax1.set_ylabel('Test error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    # ax1.set_title()
    plt.tight_layout()
    plt.legend()
    plt.xlim(-.1, 5.5)
    plt.savefig('enkf_test_error_intro.pdf',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_error_sgd_adam(accuracies, iters):
    accs = np.array(accuracies)
    plt.plot(100 - accs[:49][:, 1].astype(float), label=r'SGD $\sigma=1$')
    plt.plot(100 - accs[49:98][:, 1].astype(float), label=r'SGD $\sigma=3$')
    plt.plot(100 - accs[99:147][:, 1].astype(float), label=r'ADAM $\sigma=1$')
    plt.plot(100 - accs[147:][:, 1].astype(float), label=r'ADAM $\sigma=3$')
    plt.xlabel('Epochs')
    plt.ylabel('Test error in %')
    plt.legend()
    plt.savefig('sgd_adam_test_error.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_accuracy_all_iteration(accuracies, iters):
    fig, ax1 = plt.subplots()
    for a in accuracies:
        if a.lower().startswith('adam'):
            l = torch.load(a)
            l.insert(0, 0)
            adam = 100 - np.array(l)
            print(adam)
        elif a.lower().startswith('sgd'):
            l = torch.load(a)
            l.insert(0, 0)
            sgd = 100 - np.array(l)
        elif a.lower().startswith('acc'):
            l = torch.load(a)[0]
            l.insert(0, 0)
            enkf = 100 - np.array(l)
    ax1.plot(adam[:len(enkf)], '.-', label='ADAM')
    ax1.plot(sgd[:len(enkf)], '.-', label='SGD')
    ax1.plot(enkf, '.-', label='EnKF')
    ax1.set_ylabel('Test Error in %')
    ax1.set_xlabel('Iterations')
    plt.xticks(range(len(iters)), iters)
    plt.legend()
    tkl = ax1.xaxis.get_ticklabels()
    for label in tkl[::2]:
        label.set_visible(False)
    fig.savefig('all_test_error_iteration.eps',
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    path = '.'
    conv_params = {}
    files = []
    test_loss = []
    test_accuracy = []
    # for file in os.listdir(path):
    #     if file.startswith('conv_params_'):
    #         files.append(file)
    # files = sorted(files, key=lambda x: int(x.strip('conv_params .pt')))
    # f = os.path.join(path, 'acc_loss.pt')
    # if os.path.exists(f):
    #     losses = torch.load(f)
    #     test_accuracy = losses[0]
    #     test_loss = losses[1]
    # else:
    #     conv_net = ConvNet()
    #     criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     batch = 64
    #     train_loader, test_loader = get_data(batch, device)
    #     # conv_params = torch.load('conv_params_7000.pt', map_location='cpu')
    #     for file in files:
    #         print(file)
    #         conv_params = torch.load(file, map_location='cpu')
    #         ensemble = conv_params['ensemble'].mean(0)
    #         ensemble = torch.from_numpy(ensemble)
    #         ds = shape_parameter_to_conv_net(conv_net, ensemble)
    #         conv_net.set_parameter(ds)
    #         tsa, tsl = test(conv_net, test_loader)
    #         test_accuracy.append(tsa)
    #         test_loss.append(tsl)
    #     torch.save((test_accuracy, test_loss), 'acc_loss.pt')
    # iters = [int(f.strip('conv_params .pt')) for f in files]
    # plot_loss(test_loss, iters)
    # plot_accuracy(test_accuracy, (range(0, 8000, 500)))
    # plot_different_accuracies(range(0, 8000, 500))
    # plot_act_func_accuracies(range(0, 8000, 500))
    # plot_error_sgd_adam(torch.load('test_acc.pt'))
    # plot_test_error(torch.load('acc_loss.pt')[0], range(0, 8000, 500))
    accuracies = ['Adam_test_accuracy_iteration.pt',
                  'SGD_test_accuracy_iteration.pt', 'acc_loss.pt']
    plot_accuracy_all_iteration(accuracies, range(0, 8500, 500))
