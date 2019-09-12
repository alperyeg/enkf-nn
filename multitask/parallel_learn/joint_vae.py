import os
import torch
import torch.utils.data
import yaml
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from multitask.data_converter import NotMNISTLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.views = Views((-1, 250))
        self.encoder_conv_block1 = self.encoder_conv_block()
        self.encoder_conv_block2 = self.encoder_conv_block()
        self.encoder_block1 = self.encoder_lin_block()
        self.encoder_block2 = self.encoder_lin_block()
        self.decoder_block1 = self.decoder_block()
        self.decoder_block2 = self.decoder_block()

        self.block1 = nn.ModuleDict({'conv': self.encoder_conv_block1,
                                     'linear': self.encoder_block1})
        self.block2 = nn.ModuleDict({'conv': self.encoder_conv_block2,
                                     'linear': self.encoder_block2})

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

    def encoder_conv_block(self):
        return nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(50, 250, kernel_size=4),
            nn.ReLU(),
            self.views,
            nn.Linear(250, 30)
        )

    @staticmethod
    def encoder_lin_block():
        return nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30),
            nn.ReLU()
        )

    @staticmethod
    def decoder_block():
        return nn.Sequential(
            nn.Linear(30, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()
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
        x1 = self.block1[choice](x1)
        x2 = self.block2[choice](x2)
        # x2 = self.encoder_conv_block2(x2)
        x1 = F.relu(self.fc_shared(x1))
        x2 = F.relu(self.fc_shared(x2))

        mu1, logvar1 = self.fc_mu1(x1), self.fc_logvar1(x1)
        z1 = self.reparameterize(mu1, logvar1)

        mu2, logvar2 = self.fc_mu2(x2), self.fc_logvar2(x2)
        z2 = self.reparameterize(mu2, logvar2)

        outs = {
            'decode1': self.decoder_block1(z1),
            'mu1': mu1,
            'logvar1': logvar1,
            'decode2': self.decoder_block2(z2),
            'mu2': mu2,
            'logvar2': logvar2,
        }
        return outs


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


def train(epoch):
    model.train()
    train_loss1 = 0
    train_loss2 = 0
    for batch_idx, (data_mnist, data_notmnist) in enumerate(
            zip(train_loader_mnist, train_loader_notmnist)):
        data_mnist = data_mnist[0].to(device)
        data_notmnist = data_notmnist[0].to(device)
        optimizer.zero_grad()

        outs = model(data_mnist, data_notmnist)
        recon_batch1 = outs['decode1']
        mu1 = outs['mu1']
        logvar1 = outs['logvar1']
        recon_batch2 = outs['decode2']
        mu2 = outs['mu2']
        logvar2 = outs['logvar2']
        # calculate losses
        # first loss 1
        loss1 = loss_function(recon_batch1, data_mnist, mu1, logvar1)
        loss1.backward()
        train_loss1 += loss1.item()
        # now loss 2
        loss2 = loss_function(recon_batch2, data_notmnist, mu2, logvar2)
        loss2.backward()
        train_loss2 += loss2.item()

        optimizer.step()
        if batch_idx % config['log-interval'] == 0:
            print('Train Epoch MNIST: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                len(data_mnist), len(train_loader_mnist.dataset),
                100. * batch_idx / len(train_loader_mnist),
                loss1.item() / len(data_mnist)))
            print('Train Epoch NotMNIST: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                len(data_notmnist), len(train_loader_notmnist.dataset),
                100. * batch_idx / len(train_loader_notmnist),
                loss2.item() / len(data_notmnist)))

    print('====> Epoch: {} Average loss MNIST: {:.4f}'.format(
          epoch, train_loss1 / len(train_loader_mnist.dataset)))
    print('====> Epoch: {} Average loss NotMNIST: {:.4f}'.format(
          epoch, train_loss2 / len(train_loader_mnist.dataset)))


def test(epoch):
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    with torch.no_grad():
        for i, (data_mnist, data_notmnist) in enumerate(
                zip(test_loader_mnist, test_loader_notmnist)):
            data_mnist = data_mnist[0].to(device)
            data_notmnist = data_notmnist[0].to(device)
            outs = model(data_mnist, data_notmnist)

            recon_batch1 = outs['decode1']
            mu1 = outs['mu1']
            logvar1 = outs['logvar1']

            recon_batch2 = outs['decode2']
            mu2 = outs['mu2']
            logvar2 = outs['logvar2']

            test_loss1 += loss_function(recon_batch1,
                                        data_mnist, mu1, logvar1).item()
            test_loss2 += loss_function(recon_batch2, data_notmnist, mu2,
                                        logvar2).item()
            if i == 0:
                n = min(data_mnist.size(0), 8)
                comparison1 = torch.cat([data_mnist[:n],
                                         recon_batch1.view(config['batch_size'], 1, 28, 28)[:n]])
                n = min(data_notmnist.size(0), 8)
                comparison2 = torch.cat([data_notmnist[:n],
                                         recon_batch2.view(
                                             config['batch_size'], 1, 28, 28)[
                                         :n]])
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
    print('====> Test set loss: {:.4f}'.format(test_loss1))

    test_loss2 /= len(test_loader_mnist.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss2))


if __name__ == "__main__":
    for file in os.listdir(config['results_dir']):
        file_path = os.path.join(config['results_dir'], file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    print('deleted contents of {}'.format(config['results_dir']))
    for epoch in range(1, config['epochs'] + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(128, 30).to(device)
            sample = model.decoder_block1(sample).cpu()
            save_image(sample.view(128, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
