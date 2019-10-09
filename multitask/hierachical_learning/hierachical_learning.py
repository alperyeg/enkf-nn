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


def _print_outs_binary(epoch, num=0, batch_idx=0, len_data=0, binary_loss=0,
                       print_type="Train", correct=0):
    if print_type == "train":
        print('Binary Train {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            num, epoch, batch_idx * len_data,
            len(train_loader_mnist.dataset),
                        100. * batch_idx / len(train_loader_mnist),
                        binary_loss.item() / len_data))
        iteration = batch_idx * len_data + (
                (epoch - 1) * len(train_loader_mnist.dataset))
        writer.add_scalar('binary_loss {}'.format(num),
                          binary_loss.item(), iteration)
    elif print_type == 'epoch_train':
        if num == 0:
            mnist = 'MNIST'
        else:
            mnist = 'NotMNIST'
        print('====> Epoch: {} Average binary loss {}: {:.4f}'.format(mnist,
                                                                      epoch,
                                                                      binary_loss / len(
                                                                          train_loader_mnist.dataset)))
        writer.add_scalar('average loss {}'.format(num),
                          binary_loss / len(train_loader_mnist.dataset),
                          epoch)
    elif print_type == 'test':
        print('====> Test set loss {}: {:.4f}'.format(num, binary_loss))
        writer.add_scalar('test loss {}'.format(num), binary_loss, epoch)

    elif print_type == 'test_epoch':
        print(
            '\nTest set {}: Average loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                num, binary_loss, correct,
                len(test_loader_notmnist.dataset),
                100. * correct / len(test_loader_notmnist.dataset)))
        writer.add_scalar('Accuracy {}'.format(num), 100. * correct /
                          len(test_loader_notmnist.dataset), epoch)


def binary_train(epoch):
    binary_model.train()
    train_loss1 = 0
    train_loss2 = 0
    for batch_idx, (data_mnist, data_notmnist) in enumerate(
            zip(train_loader_mnist, train_loader_notmnist)):
        data_mnist = data_mnist[0].to(device)
        data_notmnist = data_notmnist[0].to(device)
        data = (data_mnist, data_notmnist)

        binary_optimizer.zero_grad()
        randint = torch.randint(0, 2, (1, )).item()
        x1 = binary_model(data[randint])
        binary_loss1 = loss_binary_decision(x1, randint)
        binary_loss1.backward()
        train_loss1 += binary_loss1.item()

        neg_randint = 1 - randint
        x2 = binary_model(data[neg_randint])
        binary_loss2 = loss_binary_decision(x2, neg_randint)
        binary_loss2.backward()
        train_loss2 += binary_loss2.item()

        binary_optimizer.step()

        if batch_idx % config['log-interval'] == 0:
            _print_outs_binary(epoch=epoch, num=1, batch_idx=batch_idx,
                               len_data=len(data_mnist), binary_loss=binary_loss1,
                               print_type='train')

            _print_outs_binary(epoch=epoch, num=2, batch_idx=batch_idx,
                               len_data=len(data_notmnist), binary_loss=binary_loss2,
                               print_type='train')

    _print_outs_binary(epoch=epoch, binary_loss=train_loss1, num=0,
                       print_type='train_epoch')
    _print_outs_binary(epoch=epoch, binary_loss=train_loss2, num=1,
                       print_type='train_epoch')


def binary_test(epoch):
    binary_model.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for i, (data_mnist, data_notmnist) in enumerate(
                zip(test_loader_mnist, test_loader_notmnist)):
            data_mnist = data_mnist[0].to(device)
            data_notmnist = data_notmnist[0].to(device)
            data = (data_mnist, data_notmnist)
            randint = torch.randint(0, 2, (1,)).item()
            x1 = binary_model(data[randint])
            test_loss1 += loss_binary_decision(x1, randint)

            neg_randint = 1 - randint
            x2 = binary_model(data[neg_randint])

            test_loss2 += loss_binary_decision(x2, neg_randint)
            if randint == 0:
                correct1 += x1.argmax(1).sum().item()
                correct2 += x2.argmin(1).sum().item()
            else:
                correct1 += x1.argmin(1).sum().item()
                correct2 += x2.argmax(1).sum().item()
    test_loss1 /= len(test_loader_mnist.dataset)
    _print_outs_binary(num=randint, binary_loss=test_loss1, epoch=epoch, print_type='test')

    _print_outs_binary(num=randint, correct=correct1, binary_loss=test_loss1,
                       epoch=epoch, print_type='test_epoch')

    # TODO finish the print outs
    test_loss2 /= len(test_loader_notmnist.dataset)
    print('====> Test set loss {}: {:.4f}'.format(neg_randint, test_loss2))
    writer.add_scalar('test loss {}'.format(neg_randint), test_loss2, epoch)

    print(
        '\nTest set {}: Average loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            neg_randint, test_loss2, correct2,
            len(test_loader_notmnist.dataset),
            100. * correct2 / len(test_loader_notmnist.dataset)))
    writer.add_scalar('Accuracy 2', 100. * correct2 /
                      len(test_loader_notmnist.dataset), epoch)


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


def eval_all(epoch):
    correct = 0
    with torch.no_grad():
        for i, (data_mnist, data_notmnist) in enumerate(
                zip(test_loader_mnist, test_loader_notmnist)):
            data = (data_mnist, data_notmnist)

            randint = torch.randint(0, 2, (1, ))
            decision = binary_model(data[randint][0])
            condition = bool(decision.argmax(1).sum().item() > decision.argmin(1).sum().item())
            if condition:
                x = cln_model1(data[randint][0])
            else:
                x = cln_model2(data[randint][0])
            pred = x.argmax(dim=1, keepdim=True)
            correct += pred.eq(data[randint][1].view_as(pred)).sum().item()
        print('Correct decisions: {:.0f}%'.format(100. * correct / len(test_loader_notmnist.dataset)))


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
        path = os.path.join(config['results_dir'],
                            'cln_model_mnist_ep{}.pt'.format(config['epochs']))
        cln_model1.load_state_dict(torch.load(path))
        # load classificator model 2 for notmnist
        path = os.path.join(config['results_dir'],
                            'cln_model_notmnist_ep{}.pt'.format(
                                config['epochs']))
        cln_model2.load_state_dict(torch.load(path))

    for ep in range(1, config['epochs'] + 1):
        # training & test
        binary_train(ep)
        binary_test(ep)
        classification_train(ep)
        classification_test()
        eval_all(ep)
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
