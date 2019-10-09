import os
import torch
import torch.utils.data
import yaml
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from data_converter import NotMNISTLoader
from enum import Enum

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


class Views(nn.Module):
    def __init__(self, shape):
        super(Views, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class JVAE(nn.Module):
    def __init__(self):
        super(JVAE, self).__init__()
        # sharing layer
        # TODO: check if shared layer can go into the encoder blocks
        self.fc_shared = nn.Linear(30, 250)
        # Layer for classification
        self.classification_out1 = nn.Linear(30, 10)
        self.classification_out2 = nn.Linear(30, 10)

        self.block1 = nn.ModuleDict({
            'conv_enc': self.encoder_conv_block(),
            'linear_enc': self.encoder_lin_block(),
            'conv_dec': self.decoder_conv_block(),
            'linear_dec': self.decoder_lin_block()
        })
        self.block2 = nn.ModuleDict({
            'conv_enc': self.encoder_conv_block(),
            'linear_enc': self.encoder_lin_block(),
            'conv_dec': self.decoder_conv_block(),
            'linear_dec': self.decoder_lin_block()
        })

        self.decision = self.binary_decision_block()

        # self.encoder_block1 = nn.Sequential(
        #     nn.Linear(784, 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 250),
        #     nn.ReLU(),
        #     # self.fc_shared,
        #     nn.ReLU()
        # )

        self.fc_mu1 = nn.Linear(250, 30)
        self.fc_logvar1 = nn.Linear(250, 30)

        self.fc_mu2 = nn.Linear(250, 30)
        self.fc_logvar2 = nn.Linear(250, 30)

    @staticmethod
    def encoder_conv_block():
        return nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(50, 250, kernel_size=4),
            nn.SELU(),
            Views((-1, 250)),
            nn.Linear(250, 30)

            # nn.Conv2d(1, 16, kernel_size=6, stride=2),
            # nn.SELU(),
            # nn.Conv2d(16, 32, kernel_size=4, stride=2),
            # nn.SELU(),
            # nn.Conv2d(32, 64, kernel_size=2, stride=2),
            # nn.SELU(),
            # Views((-1, 256)),
            # nn.Linear(256, 30)
        )

    @staticmethod
    def encoder_lin_block():
        return nn.Sequential(
            nn.Linear(784, 1000),
            nn.SELU(),
            nn.Linear(1000, 500),
            nn.SELU(),
            nn.Linear(500, 250),
            nn.SELU(),
            nn.Linear(250, 30),
            nn.LeakyReLU()
        )

    @staticmethod
    def decoder_lin_block():
        return nn.Sequential(
            nn.Linear(30, 250),
            nn.SELU(),
            nn.Linear(250, 500),
            nn.SELU(),
            nn.Linear(500, 1000),
            nn.SELU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()
        )

    @staticmethod
    def decoder_conv_block():
        return nn.Sequential(
            nn.Linear(30, 64*2*2),
            nn.SELU(),
            Views((-1, 64, 2, 2)),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.SELU(),
            # nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=3, padding=1),
            nn.SELU(),
            # nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=6, stride=3, padding=4),
            nn.SELU(),
            nn.Sigmoid()
        )

    @staticmethod
    def binary_decision_block():
        return nn.Sequential(
            nn.Linear(30, 250),
            nn.SELU(),
            nn.Linear(250, 100),
            nn.SELU(),
            nn.Linear(100, 2),
            nn.Sigmoid(),
        )

    # def encode(self, x):
    #     h = F.relu(self.fc1(x))
    #     return self.fc21(h), self.fc22(h)
    #

    # def decode(self, z):
    #     h3 = F.relu(self.fc3(z))
    #     return torch.sigmoid(self.fc4(h3))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1, x2, choice='conv'):
        if choice == 'linear':
            x1 = x1.view(-1, 784)
            x2 = x2.view(-1, 784)
        # x1 = self.encoder_block1(x1.view(-1, 784))
        # x2 = self.encoder_block2(x2.view(-1, 784))

        # x1 = self.encoder_conv_block1(x1)
        x1 = self.block1[choice + '_enc'](x1)
        x2 = self.block2[choice + '_enc'](x2)
        # x2 = self.encoder_conv_block2(x2)
        x1 = F.relu(self.fc_shared(x1))
        x2 = F.relu(self.fc_shared(x2))

        mu1, logvar1 = self.fc_mu1(x1), self.fc_logvar1(x1)
        z1 = self.reparameterize(mu1, logvar1)

        mu2, logvar2 = self.fc_mu2(x2), self.fc_logvar2(x2)
        z2 = self.reparameterize(mu2, logvar2)

        z_s = (z1.clone().detach().requires_grad_(True),
               z2.clone().detach().requires_grad_(True))

        outs = {
            'decode1': self.block1['linear_dec'](z1),
            'mu1': mu1,
            'logvar1': logvar1,
            'decode2': self.block2['linear_dec'](z2),
            'mu2': mu2,
            'logvar2': logvar2,
            'classification1': F.relu(self.classification_out1(z1)),
            'classification2': F.relu(self.classification_out2(z2)),
            'binary_decision1': F.relu(self.decision(z_s[0])),
            'binary_decision2': F.relu(self.decision(z_s[1]))
        }
        return outs


class LossFunctions(Enum):
    BCE = 'BCE'
    CE = 'CE'
    MAE = 'MAE'
    MSE = 'MSE'
    NORM = 'NORM'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


model = JVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


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


def loss_function_classification(recon_x, x, mu, logvar, output, target):
    loss_r = loss_function(recon_x, x, mu, logvar)
    loss_c = F.cross_entropy(output, target, reduction='sum')
    # loss_b = loss_binary_decision(decision, randint)
    return loss_c + loss_r


def train(epoch):
    model.train()
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0.
    train_loss4 = 0
    loss1_0 = 0.
    loss2_0 = 0.
    scaling_value = torch.as_tensor(
        (0.4171190549233725 / epoch), dtype=torch.float32, device=device)
    for batch_idx, (data_mnist, data_notmnist) in enumerate(
            zip(train_loader_mnist, train_loader_notmnist)):
        labels_mnist = data_mnist[1].to(device)
        data_mnist = data_mnist[0].to(device)
        labels_notmnist = data_notmnist[1].to(device)
        data_notmnist = data_notmnist[0].to(device)
        optimizer.zero_grad()

        # rand_i = torch.randint(0, 2, (1, ))
        outs = model(data_mnist, data_notmnist)
        recon_batch1 = outs['decode1']
        mu1 = outs['mu1']
        logvar1 = outs['logvar1']
        recon_batch2 = outs['decode2']
        mu2 = outs['mu2']
        logvar2 = outs['logvar2']
        decision = outs['binary_decision1']
        binary_loss1 = loss_binary_decision(decision, 0)
        binary_loss1.backward()
        train_loss3 += binary_loss1.item()

        decision = outs['binary_decision2']
        binary_loss2 = loss_binary_decision(decision, 1)
        binary_loss2.backward()
        train_loss4 += binary_loss2.item()

        # calculate losses
        # first loss 1
        # loss1 = loss_function(recon_batch1, data_mnist, mu1, logvar1)
        loss1 = loss_function_classification(recon_batch1, data_mnist,
                                             mu1, logvar1,
                                             outs['classification1'],
                                             labels_mnist)
        # if epoch <= 2:
        #     loss1 = scaling_value * loss1

        loss1.backward()
        train_loss1 += loss1.item()
        # now loss 2
        # loss2 = loss_function(recon_batch2, data_notmnist, mu2, logvar2)
        loss2 = loss_function_classification(recon_batch2, data_notmnist, mu2,
                                             logvar2, outs['classification2'],
                                             labels_notmnist)

        if epoch <= 2:
            loss2 = scaling_value * loss2

        loss2.backward()
        train_loss2 += loss2.item()

        if loss1_0 + loss2_0 == 0.:
            loss1_0 = loss1.item()
            loss2_0 = loss2.item()

        optimizer.step()

        if batch_idx % config['log-interval'] == 0:
            print('Train Epoch MNIST: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                len(data_mnist), len(train_loader_mnist.dataset),
                100. * batch_idx / len(train_loader_mnist),
                loss1.item() / len(data_mnist)))
            iteration = batch_idx * \
                len(data_mnist) + ((epoch - 1) *
                                   len(train_loader_mnist.dataset))
            writer.add_scalar('train_loss 1', loss1.item(), iteration)
            print('Train Epoch NotMNIST: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_notmnist),
                len(train_loader_notmnist.dataset),
                100. * batch_idx / len(train_loader_notmnist),
                loss2.item() / len(data_notmnist)))
            print('Binary loss 1 {}, set {}'.format(binary_loss1.item(), 0))
            print('Binary loss 2 {}, set {}'.format(binary_loss2.item(), 1))
            writer.add_scalar('Binary loss 1', binary_loss1.item(), iteration)
            writer.add_scalar('Binary loss 2', binary_loss2.item(), iteration)
            # print('Classification loss {}'.format(loss1.item()))
            print('scaling loss: l1 {} l2 {} total {}'.format(
                loss1.item() / loss1_0, loss2.item() / loss2_0,
                (loss1.item() + loss2.item()) / (loss1_0 + loss2_0)))
            writer.add_scalar('Classification loss 1', loss1.item(), iteration)
            writer.add_scalar('Classification loss 2', loss2.item(), iteration)

    print('====> Epoch: {} Average loss MNIST: {:.4f}'.format(
          epoch, train_loss1 / len(train_loader_mnist.dataset)))
    writer.add_scalar('average loss 1',
                      train_loss1 / len(train_loader_mnist.dataset), epoch)
    print('====> Epoch: {} Average loss NotMNIST: {:.4f}'.format(
          epoch, train_loss2 / len(train_loader_mnist.dataset)))
    writer.add_scalar('average loss 2',
                      train_loss2 / len(train_loader_mnist.dataset), epoch)
    print(
        'Average Binary loss {}'.format(train_loss3 / len(train_loader_mnist.dataset)))
    print(
        'Average Binary loss {}'.format(train_loss4 / len(train_loader_mnist.dataset)))
    writer.add_scalar('Average binary loss 1',
                      train_loss3 / len(train_loader_mnist.dataset), epoch)
    writer.add_scalar('Average binary loss 2',
                      train_loss4 / len(train_loader_notmnist.dataset), epoch)


def test(epoch):
    print('---- Test ----')
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    test_loss3 = 0
    test_loss4 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    with torch.no_grad():
        for i, (data_mnist, data_notmnist) in enumerate(
                zip(test_loader_mnist, test_loader_notmnist)):
            labels_mnist = data_mnist[1].to(device)
            data_mnist = data_mnist[0].to(device)
            labels_notmnist = data_notmnist[1].to(device)
            data_notmnist = data_notmnist[0].to(device)
            outs = model(data_mnist, data_notmnist)

            recon_batch1 = outs['decode1']
            mu1 = outs['mu1']
            logvar1 = outs['logvar1']

            recon_batch2 = outs['decode2']
            mu2 = outs['mu2']
            logvar2 = outs['logvar2']

            # test_loss1 += loss_function(recon_batch1,
            #                             data_mnist, mu1, logvar1).item()
            # test_loss2 += loss_function(recon_batch2, data_notmnist, mu2,
            #                             logvar2).item()
            test_loss1 += loss_function_classification(recon_batch1,
                                                       data_mnist,
                                                       mu1, logvar1,
                                                       outs['classification1'],
                                                       labels_mnist)
            test_loss2 += loss_function_classification(recon_batch2,
                                                       data_notmnist,
                                                       mu2, logvar2,
                                                       outs['classification2'],
                                                       labels_notmnist)

            test_loss3 += loss_binary_decision(outs['binary_decision1'], 0)

            test_loss4 += loss_binary_decision(outs['binary_decision2'], 1)

            # get the index of the max log-probability
            pred = outs['classification1'].argmax(dim=1, keepdim=True)
            correct1 += pred.eq(labels_mnist.view_as(pred)).sum().item()

            pred = outs['classification2'].argmax(dim=1, keepdim=True)
            correct2 += pred.eq(labels_notmnist.view_as(pred)).sum().item()

            pred = outs['binary_decision1']
            # correct3 += pred.argmax(1).sum().item()
            correct3 += (len(pred) - pred.argmin(dim=1).sum()).item()

            pred = outs['binary_decision2']
            # correct4 += pred.argmin(1).sum().item()
            correct4 += (len(pred) - pred.argmax(dim=1).sum()).item()

            if i == 0:
                n = min(data_mnist.size(0), 8)
                comparison1 = torch.cat([data_mnist[:n], recon_batch1.view(
                    config['batch_size'], 1, 28, 28)[:n]])
                n = min(data_notmnist.size(0), 8)
                comparison2 = torch.cat([data_notmnist[:n], recon_batch2.view(
                    config['batch_size'], 1, 28, 28)[:n]])
                results_dir = config['results_dir']
                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)

                save_image(comparison1.cpu(),
                           os.path.join(results_dir, 'reconstruction_mnist_' +
                                        str(epoch) + '.png'), nrow=n)
                save_image(comparison2.cpu(),
                           os.path.join(results_dir,
                                        'reconstruction_notmnist_' + str(
                                            epoch) + '.png'), nrow=n)
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

    test_loss3 /= len(test_loader_mnist.dataset)
    print('====> Binary loss 1: {:.4f}'.format(test_loss3))
    writer.add_scalar('test loss 3', test_loss3, epoch)

    print('Test set average decision accuracy for set 0 {}'.format(
        correct3 / len(test_loader_mnist.dataset)))
    writer.add_scalar('Accuracy 3',
                      100 * correct3 / len(test_loader_mnist.dataset), epoch)

    test_loss4 /= len(test_loader_notmnist.dataset)
    print('====> Binary loss 2: {:.4f}'.format(test_loss4))
    writer.add_scalar('test loss 4', test_loss4, epoch)

    print('Test set average decision accuracy for set 1 {}'.format(
        100 * correct4 / len(test_loader_notmnist.dataset)))
    writer.add_scalar('Accuracy 4',
                      correct4 / len(test_loader_notmnist.dataset), epoch)


if __name__ == "__main__":
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
        path = os.path.join(config['results_dir'],
                            'model_ep{}.pt'.format(config['epochs']))
        model.load_state_dict(torch.load(path))
    for epoch in range(1, config['epochs'] + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(128, 30).to(device)
            sample = model.block1['linear_dec'](sample).cpu()
            save_image(sample.view(128, 1, 28, 28),
                       'results/sample_mnist' + str(epoch) + '.png')
            sample = torch.randn(128, 30).to(device)
            sample = model.block2['linear_dec'](sample).cpu()
            save_image(sample.view(128, 1, 28, 28),
                       'results/sample_notmnist_' + str(epoch) + '.png')

        torch.save(model.state_dict(),
                   os.path.join(config['results_dir'],
                                'model_ep{}.pt'.format(config['epochs'])))
writer.close()
