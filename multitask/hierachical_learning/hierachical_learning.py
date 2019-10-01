import os
import torch
import torch.utils.data
import yaml
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from data_converter import NotMNISTLoader
from models import BinaryNet, ClassificatorNet, loss_binary_decision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(write_to_disk=False)

with open('config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
        print(config)
    except yaml.YAMLError as err:
        print(err)


torch.manual_seed(config['manualSeed'])

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
train_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST(config['dataroot'], train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=config['batch_size'], shuffle=True, **kwargs)
test_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST(config['dataroot'], train=False,
                   transform=transforms.ToTensor()),
    batch_size=config['batch_size'], shuffle=True, **kwargs)
not_mnist = NotMNISTLoader(folder_path=config['folder_path'])
if config['create_dataloader']:
    train_loader_notmnist, test_loader_notmnist = not_mnist.create_dataloader(
        batch_size=config['batch_size'], save=True, test_size=10000,
        train_size=60000,
        **{'filename': config['filename']})
else:
    not_mnist_dict = not_mnist.load_from_file(config['filename'])
    train_loader_notmnist = not_mnist_dict['train_loader']
    test_loader_notmnist = not_mnist_dict['test_loader']


binary_model = BinaryNet().to(device)
binary_optimizer = optim.Adam(binary_model.parameters(), lr=1e-3)
# mnist model
cln_model1 = ClassificatorNet().to(device)
cln1_optimizer = optim.Adam(cln_model1.parameters(), lr=1e-3)
# notmnist model
cln_model2 = ClassificatorNet().to(device)
cln2_optimizer = optim.Adam(cln_model2.parameters(), lr=1e-3)


def binary_train(epoch):
    binary_model.train()
    train_loss1 = 0
    train_loss2 = 0
    for batch_idx, (data_mnist, data_notmnist) in enumerate(
            zip(train_loader_mnist, train_loader_notmnist)):
        # labels_mnist = data_mnist[1].to(device)
        data_mnist = data_mnist[0].to(device)
        # labels_notmnist = data_notmnist[1].to(device)
        data_notmnist = data_notmnist[0].to(device)
        binary_optimizer.zero_grad()

        x1, x2 = binary_model(data_mnist, data_notmnist)
        binary_loss1 = loss_binary_decision(x1, 0)
        binary_loss1.backward()
        train_loss1 += binary_loss1.item()
        binary_loss2 = loss_binary_decision(x2, 1)
        binary_loss2.backward()
        train_loss2 += binary_loss2.item()

        binary_optimizer.step()

        if batch_idx % config['log-interval'] == 0:
            print('Binary Train Epoch MNIST: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_mnist), len(train_loader_mnist.dataset),
                100. * batch_idx / len(train_loader_mnist),
                binary_loss1.item() / len(data_mnist)))
            iteration = batch_idx * len(data_mnist) + (
                        (epoch - 1) * len(train_loader_mnist.dataset))
            writer.add_scalar('binary_loss 1', binary_loss1.item(), iteration)

            print('Binary Train Epoch NotMNIST: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_notmnist), len(train_loader_notmnist.dataset),
                100. * batch_idx / len(train_loader_notmnist),
                binary_loss2.item() / len(data_notmnist)))
            iteration = batch_idx * len(data_notmnist) + (
                        (epoch - 1) * len(train_loader_notmnist.dataset))
            writer.add_scalar('binary_loss 2', binary_loss2.item(), iteration)

    print('====> Epoch: {} Average binary loss MNIST: {:.4f}'.format(
        epoch, train_loss1 / len(train_loader_mnist.dataset)))
    writer.add_scalar('average loss 1',
                      train_loss1 / len(train_loader_mnist.dataset), epoch)
    print('====> Epoch: {} Average binary loss NotMNIST: {:.4f}'.format(
        epoch, train_loss2 / len(train_loader_mnist.dataset)))
    writer.add_scalar('average loss 1',
                      train_loss2 / len(train_loader_notmnist.dataset), epoch)


def classification_train(epoch):
    cln_model1.train()
    cln_model2.train()
    # train_loss1 = 0
    # train_loss2 = 0
    for batch_idx, (data_mnist, data_notmnist) in enumerate(
            zip(train_loader_mnist, train_loader_notmnist)):
        labels_mnist = data_mnist[1].to(device)
        data_mnist = data_mnist[0].to(device)
        labels_notmnist = data_notmnist[1].to(device)
        data_notmnist = data_notmnist[0].to(device)

        # Classifier 1
        cln1_optimizer.zero_grad()
        x1 = cln_model1(data_mnist)
        loss1 = F.nll_loss(x1, labels_mnist)
        loss1.backward()
        cln1_optimizer.step()

        # Classifier 2
        cln2_optimizer.zero_grad()
        x2 = cln_model2(data_notmnist)
        loss2 = F.nll_loss(x2, labels_notmnist)
        loss2.backward()
        cln2_optimizer.step()
        if batch_idx % config['log-interval'] == 0:
            print('Train MNIST Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_mnist), len(train_loader_mnist.dataset),
                100. * batch_idx / len(train_loader_mnist), loss1.item()))

            print('Train NotMNISTEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_notmnist), len(train_loader_notmnist.dataset),
                100. * batch_idx / len(train_loader_notmnist), loss2.item()))


def classification_test():
    cln_model1.eval()
    cln_model2.eval()
    correct1 = 0
    correct2 = 0
    test_loss1 = 0
    test_loss2 = 0
    with torch.no_grad():
        for i, (data_mnist, data_notmnist) in enumerate(
                zip(test_loader_mnist, test_loader_notmnist)):
            labels_mnist = data_mnist[1].to(device)
            data_mnist = data_mnist[0].to(device)

            labels_notmnist = data_notmnist[1].to(device)
            data_notmnist = data_notmnist[0].to(device)

            x1 = cln_model1(data_mnist)
            test_loss1 += F.nll_loss(x1, labels_mnist, reduction='sum')
            pred = x1.argmax(dim=1, keepdim=True)
            correct1 += pred.eq(labels_mnist.view_as(pred)).sum().item()

            x2 = cln_model2(data_notmnist)
            test_loss2 += F.nll_loss(x2, labels_notmnist, reduction='sum')
            pred = x2.argmax(dim=1, keepdim=True)
            correct2 += pred.eq(labels_notmnist.view_as(pred)).sum().item()
        test_loss1 /= len(test_loader_mnist.dataset)
        test_loss2 /= len(test_loader_notmnist.dataset)

        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss1, correct1, len(test_loader_mnist.dataset),
                100. * correct1 / len(test_loader_mnist.dataset)))

        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss2, correct2, len(test_loader_notmnist.dataset),
                100. * correct2 / len(test_loader_notmnist.dataset)))


def binary_test(epoch):
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for i, (data_mnist, data_notmnist) in enumerate(
                zip(test_loader_mnist, test_loader_notmnist)):
            data_mnist = data_mnist[0].to(device)
            data_notmnist = data_notmnist[0].to(device)
            x1, x2 = binary_model(data_mnist, data_notmnist)
            test_loss1 += loss_binary_decision(x1, 0)
            test_loss2 += loss_binary_decision(x2, 0)
            correct1 += x1.argmax(1).sum().item()
            correct2 += x2.argmin(1).sum().item()
    test_loss1 /= len(test_loader_mnist.dataset)
    print('====> Test set loss 1: {:.4f}'.format(test_loss1))
    writer.add_scalar('test loss 1', test_loss1, epoch)

    print('\nTest set MNIST: Average loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss1, correct1, len(test_loader_mnist.dataset),
        100. * correct1 / len(test_loader_mnist.dataset)))
    writer.add_scalar('Accuracy 1', 100. * correct1 /
                      len(test_loader_mnist.dataset), epoch)

    test_loss2 /= len(test_loader_notmnist.dataset)
    print('====> Test set loss 2: {:.4f}'.format(test_loss2))
    writer.add_scalar('test loss 2', test_loss2, epoch)

    print('\nTest set NotMNIST: Average loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss2, correct2, len(test_loader_notmnist.dataset),
        100. * correct2 / len(test_loader_notmnist.dataset)))
    writer.add_scalar('Accuracy 2', 100. * correct2 /
                      len(test_loader_notmnist.dataset), epoch)


if __name__ == "__main__":
    if not os.path.exists(config['results_dir']):
        os.mkdir(config['results_dir'])
    for file in os.listdir(config['results_dir']):
        file_path = os.path.join(config['results_dir'], file)
        try:
            if os.path.isfile(file_path) and file_path.endswith('.png'):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    print('deleted contents of {}'.format(
        os.path.abspath(config['results_dir'])))
    # if model should be loaded
    if config['load_model']:
        # load binary model
        path = os.path.join(config['results_dir'],
                            'binary_model_ep{}.pt'.format(config['epochs']))
        binary_model.load_state_dict(torch.load(path))
        # load classificator model 1 for mnist
        # path = os.path.join(config['results_dir'],
        #                     'cln_model_mnist_ep{}.pt'.format(config['epochs']))
        # cln_model1.load_state_dict(torch.load(path))
        # # load classificator model 2 for notmnist
        # path = os.path.join(config['results_dir'],
        #                     'cln_model_notmnist_ep{}.pt'.format(
        #                         config['epochs']))
        # cln_model2.load_state_dict(torch.load(path))

    for ep in range(1, config['epochs'] + 1):
        binary_train(ep)
        binary_test(ep)
        classification_train(ep)
        classification_test()
        with torch.no_grad():
            torch.save(binary_model.state_dict(),
                       os.path.join(config['results_dir'],
                                    'binary_model_ep{}.pt'.format(
                                        config['epochs'])))
            torch.save(cln_model1.state_dict(),
                       os.path.join(config['results_dir'],
                                    'cln_model_mnist_ep{}.pt'.format(
                                        config['epochs'])))
            torch.save(cln_model2.state_dict(),
                       os.path.join(config['results_dir'],
                                    'cln_model_notmnist_ep{}.pt'.format(
                                        config['epochs'])))
writer.close()
