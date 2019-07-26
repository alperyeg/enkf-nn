import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from numpy.linalg import norm

import numpy as np
from mlp_net import MLPNet
from enkf import EnsembleKalmanFilter as EnKF
from enkf import _encode_targets
from finalPlots import Plotter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    """
    Convenience class.


    """

    def __init__(self):
        self.data_fashion = None
        self.data_mnist = None
        self.test_mnist = None
        self.test_fashion = None
        self.data_fashion_loader = None
        self.data_mnist_loader = None
        self.test_fashion_loader = None
        self.test_mnist_loader = None

    def init_iterators(self, root, batch_size):
        self.data_fashion_loader, self.data_mnist_loader, \
            self.test_fashion_loader, self.test_mnist_loader = self.load_data(
                root, batch_size)
        # here are the expensive operations
        self.data_mnist = self._make_iterator(self.data_mnist_loader)
        self.test_mnist = self._make_iterator(self.test_mnist_loader)
        self.data_fashion = self._make_iterator(self.data_fashion_loader)
        self.test_fashion = self._make_iterator(self.test_fashion_loader)

    @staticmethod
    def _make_iterator(iterable):
        """
        Return an iterator for a given iterable.
        Here `iterable` is a pytorch `DataLoader`
        """
        return iter(iterable)

    @staticmethod
    def load_data(root, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.1307], [0.3081])
             ])

        trainset_fashion = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform)
        trainloader_fashion = torch.utils.data.DataLoader(trainset_fashion,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          num_workers=2)

        testset_fashion = torchvision.datasets.FashionMNIST(root=root,
                                                            train=False,
                                                            download=True,
                                                            transform=transform)
        testload_fashion = torch.utils.data.DataLoader(testset_fashion,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=2)
        # load now MNIST dataset
        trainset_mnist = torchvision.datasets.MNIST(root=root,
                                                    train=True,
                                                    download=True,
                                                    transform=transform)
        trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=2)

        testset_mnist = torchvision.datasets.MNIST(root=root,
                                                   train=False,
                                                   download=True,
                                                   transform=transform)
        testload_mnist = torch.utils.data.DataLoader(testset_mnist,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=2)
        return trainloader_fashion, trainloader_mnist, testload_fashion, testload_mnist

    def dataiter_mnist(self):
        """ MNIST training set list iterator """
        return self.data_mnist.next()

    def dataiter_fashion(self):
        """ MNISTFashion training set list iterator """
        return self.data_fashion.next()

    def testiter_mnist(self):
        """ MNIST test set list iterator """
        return self.test_mnist.next()

    def testiter_fashion(self):
        """ MNISTFashion test set list iterator """
        return self.test_fashion.next()


class MnistFashionOptimizee:
    def __init__(self, seed, n_ensembles, batch_size, root):

        self.random_state = np.random.RandomState(seed=seed)

        self.n_ensembles = n_ensembles
        self.batch_size = batch_size
        self.root = root

        self.mlp_net = MLPNet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = DataLoader()
        self.data_loader.init_iterators(self.root, self.batch_size)
        self.dataiter_fashion = self.data_loader.dataiter_fashion
        self.dataiter_mnist = self.data_loader.dataiter_mnist
        self.testiter_fashion = self.data_loader.testiter_fashion
        self.testiter_mnist = self.data_loader.testiter_mnist

        self.generation = 0
        # if self.generation % 2 == 0:
        #     self.inputs, self.labels = self.dataiter_fashion()
        # else:
        #     self.inputs, self.labels = self.dataiter_mnist()
        self.inputs, self.labels = self.dataiter_mnist()
        self.test_input, self.test_label = self.testiter_mnist()

        # For plotting
        self.train_pred = []
        self.test_pred = []
        self.train_acc = []
        self.test_acc = []
        self.train_cost = []
        self.test_cost = []
        self.targets = []
        self.output_activity_train = []
        self.output_activity_test = []

        self.targets.append(self.labels)

    def create_individual(self):
        # get weights, biases from networks and flatten them
        # convolutional network parameters ###
        mlp_ensembles = []
        with torch.no_grad():
            length = 0
            for key in self.mlp_net.state_dict().keys():
                length += self.mlp_net.state_dict()[key].nelement()

            # mlp_ensembles.append(params)
            # l = np.random.uniform(-.5, .5, size=length)
            for _ in range(self.n_ensembles):
                #     stacked = []
                #     for key in self.conv_net.state_dict().keys():
                #         stacked.extend(
                #             self._he_init(self.conv_net.state_dict()[key]))
                #     conv_ensembles.append(stacked)
                # tmp = []
                # for j in l:
                #     jitter = np.random.uniform(-0.1, 0.1) + j
                #     tmp.append(jitter)
                # conv_ensembles.append(tmp)
                mlp_ensembles.append(np.random.uniform(-1, 1, size=length))
            return dict(mlp_params=np.array(mlp_ensembles),
                        targets=self.labels.numpy(),
                        input=self.inputs.squeeze().numpy())

    @staticmethod
    def _he_init(weights, gain=0):
        """
        He- or Kaiming- initialization as in He et al., "Delving deep into
        rectifiers: Surpassing human-level performance on ImageNet
        classification". Values are sampled from
        :math:`\\mathcal{N}(0, \\text{std})` where

        .. math::
        \text{std} = \\sqrt{\\frac{2}{(1 + a^2) \\times \text{fan\\_in}}}

        Note: Only for the case that the non-linearity of the network
            activation is `relu`

        :param weights, tensor
        :param gain, additional scaling factor, Default is 0
        :return: numpy nd array, random array of size `weights`
        """
        fan_in = torch.nn.init._calculate_correct_fan(weights, 'fan_in')
        stddev = np.sqrt(2. / fan_in * (1 + gain ** 2))
        return np.random.normal(0, stddev, weights.numel())

    def set_parameters(self, ensembles):
        # set the new parameter for the network
        mlp_params = np.mean(ensembles, axis=0)
        ds = self._shape_parameter_to_mlp_net(mlp_params)
        self.mlp_net.set_parameter(ds)
        print('---- Train -----')
        print('Generation ', self.generation)
        generation_change = 8
        with torch.no_grad():
            inputs = self.inputs
            labels = self.labels

            if self.generation % generation_change == 0:
                self.inputs, self.labels = self.dataiter_mnist()
                print('New MNIST set used at generation {}'.format(
                    self.generation))
                # append the outputs
                self.targets.append(self.labels.numpy())

            outputs = self.mlp_net(inputs)
            self.output_activity_train.append(outputs.numpy())
            mlp_loss = self.criterion(outputs, labels).item()
            train_cost = _calculate_cost(_encode_targets(labels, 10),
                                         outputs.numpy(), 'MSE')
            train_acc = score(labels.numpy(), np.argmax(outputs.numpy(), 1))
            print('Cost: ', train_cost)
            print('Accuracy: ', train_acc)
            print('Loss:', mlp_loss)
            self.train_cost.append(train_cost)
            self.train_acc.append(train_acc)
            self.train_pred.append(np.argmax(outputs.numpy(), 1))

            print('---- Test -----')
            test_output = self.mlp_net(self.test_input)
            test_output = test_output.numpy()
            test_acc = score(self.test_label.numpy(),
                             np.argmax(test_output, 1))
            test_cost = _calculate_cost(_encode_targets(self.test_label, 10),
                                        test_output, 'MSE')
            print('Test accuracy', test_acc)
            self.test_acc.append(test_acc)
            self.test_pred.append(np.argmax(test_output, 1))
            self.test_cost.append(test_cost)
            self.output_activity_test.append(test_output)
            print('-----------------')
            mlp_params = []
            for c in ensembles:
                ds = self._shape_parameter_to_mlp_net(c)
                self.mlp_net.set_parameter(ds)
                mlp_params.append(self.mlp_net(inputs).numpy().T)

            outs = {
                'mlp_params': np.array(mlp_params),
                'mlp_loss': float(mlp_loss),
                'input': self.inputs.squeeze().numpy(),
                'targets': self.labels.numpy()
            }
            # if self.generation % generation_change == 0:
            # min_ens = ensembles.min()
            # max_ens = ensembles.max()
            # len_ens = len(ensembles.mean(0))
            # ens_mean = ensembles.mean(0)
            # ens_std = ensembles.std(0)
            # for _ in range(100):
            #     mlp_params.append(
            #         # ensembles.mean(0) + np.random.uniform(min_ens, max_ens,
            #         #                                       len_ens))
            #         ens_mean + np.random.normal(ens_mean, ens_std,
            #                                     size=ensembles.shape))
        return outs

    def _shape_parameter_to_mlp_net(self, params):
        param_dict = dict()
        start = 0
        for key in self.mlp_net.state_dict().keys():
            shape = self.mlp_net.state_dict()[key].shape
            length = self.mlp_net.state_dict()[key].nelement()
            end = start + length
            param_dict[key] = params[start:end].reshape(shape)
            start = end
        return param_dict


def _calculate_cost(y, y_hat, loss_function='CE'):
    """
    Loss functions
    :param y: target
    :param y_hat: calculated output (here G(u) or feed-forword output a)
    :param loss_function: name of the loss function
           `MAE` is the Mean Absolute Error or l1 - loss
           `MSE` is the Mean Squared Error or l2 - loss
           `CE` cross-entropy loss, requires `y_hat` to be in [0, 1]
           `norm` norm-2 or Forbenius norm of `y - y_hat`
    :return: cost calculated according to `loss_function`
    """
    if loss_function == 'CE':
        term1 = -y * np.log(y_hat)
        term2 = (1 - y) * np.log(1 - y_hat)
        return np.sum(term1 - term2)
    elif loss_function == 'MAE':
        return np.sum(np.absolute(y_hat - y)) / len(y)
    elif loss_function == 'MSE':
        return np.sum((y_hat - y) ** 2) / len(y)
    elif loss_function == 'norm':
        return norm(y - y_hat)
    else:
        raise KeyError(
            'Loss Function \'{}\' not understood.'.format(loss_function))


def score(x, y):
    """
    :param x: targets
    :param y: prediction
    :return:
    """
    print('target ', x)
    print('predict ', y)
    n_correct = np.count_nonzero(y == x)
    n_total = len(y)
    sc = n_correct / n_total
    return sc


def plot_distributions(dist, title='Distribution'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Digits')
    ax.set_ylabel('Frequency')
    ax.hist(dist)
    plt.show()


def jitter_ensembles(ens, ens_size):
    res = []
    for _ in range(ens_size):
        res.append(ens + np.random.uniform(-0.01, 0.01, len(ens)))
    return np.array(res)


if __name__ == '__main__':
    root = '/home/yegenoglu/Documents/toolbox/L2L/l2l/optimizees/multitask'
    n_ensembles = 1000
    mlp_loss_mnist = []
    np.random.seed(0)
    batch_size = 32
    model = MnistFashionOptimizee(root=root, batch_size=batch_size, seed=0,
                                  n_ensembles=n_ensembles)
    mlp_ens = None
    gamma = np.eye(10) * 0.001
    enkf = EnKF(tol=1e-5,
                maxit=1,
                stopping_crit='',
                online=False,
                shuffle=False,
                n_batches=1,
                converge=False)
    for i in range(100):
        model.generation = i + 1
        if i == 0:
            out = model.create_individual()
            mlp_ens = out['mlp_params']
            out = model.set_parameters(mlp_ens)
            print('loss {} generation {}'.format(out['mlp_loss'],
                                                 model.generation))
        t = time.time()
        enkf.fit(data=out['input'],
                 ensemble=mlp_ens,
                 ensemble_size=n_ensembles,
                 moments1=np.mean(mlp_ens, axis=0),
                 observations=out['targets'],
                 model_output=out['mlp_params'], noise=0.0,
                 gamma=gamma, p=None)
        print('done in {} s'.format(time.time() - t))
        mlp_ens = enkf.ensemble
        out = model.set_parameters(mlp_ens)
        mlp_loss_mnist.append(out['mlp_loss'])
    param_dict = {
        'train_pred': model.train_pred,
        'test_pred': model.test_pred,
        'train_acc': model.train_acc,
        'test_acc': model.test_acc,
        'train_cost': model.train_cost,
        'test_cost': model.test_cost,
        'train_targets': model.targets,
        'train_act': model.output_activity_train,
        'test_act': model.output_activity_test,
        'test_targets': model.test_label,
        'ensemble': mlp_ens,
    }
    np.save('mlp_params.npy', param_dict)
    # d = {
    #     'total_cost': {'total_cost': mlp_loss_mnist}
    # }
    # plotter = Plotter(d)
    # plotter.plot()
    # plotter.get_plots()
