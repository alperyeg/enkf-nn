import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time

from data_converter import NotMNISTLoader
from conv_net import ConvNet
from enkf import EnsembleKalmanFilter as EnKF
from enkf import _encode_targets
from numpy.linalg import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    """
    Convenience class.

    """

    def __init__(self):
        self.data_mnist = None
        self.test_mnist = None
        self.data_notmnist = None
        self.test_notmnist = None

        self.data_mnist_loader = None
        self.test_mnist_loader = None

        self.train_notmnist_loader = None
        self.test_notmnist_loader = None

        self.notmnist = NotMNISTLoader()

    def init_iterators(self, root, path, batch_size):
        self.data_mnist_loader, self.test_mnist_loader = self.load_data(root,
                                                                        batch_size)
        try:
            dataloader = self.notmnist.load_from_file(path)
            self.train_notmnist_loader = dataloader['train_loader']
            self.test_notmnist_loader = dataloader['test_loader']
        except FileNotFoundError:
            self.train_notmnist_loader, self.notmnist \
                = self.notmnist.create_dataloader(batch_size, save=True,
                                                  **{'filename': 'dataloader.npy'})
        # create iterators
        self.data_mnist = self._make_iterator(self.data_mnist_loader)
        self.test_mnist = self._make_iterator(self.test_mnist_loader)
        self.data_notmnist = self._make_iterator(self.train_notmnist_loader)
        self.test_notmnist = self._make_iterator(self.test_notmnist_loader)

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
        return trainloader_mnist, testload_mnist

    def dataiter_mnist(self):
        """ MNIST training set list iterator """
        return self.data_mnist.next()

    def testiter_mnist(self):
        """ MNIST test set list iterator """
        return self.test_mnist.next()

    def dataiter_notmnist(self):
        """ NotMNIST training set list iterator """
        return self.data_notmnist.next()

    def testiter_notmnist(self):
        """ NotMNIST test set list iterator """
        return self.test_notmnist.next()


class MnistOptimizee:
    def __init__(self, seed, n_ensembles, batch_size, root, path):

        self.random_state = np.random.RandomState(seed=seed)

        self.n_ensembles = n_ensembles
        self.batch_size = batch_size
        self.root = root
        self.path = path

        self.conv_net = ConvNet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = DataLoader()
        self.data_loader.init_iterators(self.root, self.path, self.batch_size)
        self.dataiter_mnist = self.data_loader.dataiter_mnist
        self.testiter_mnist = self.data_loader.testiter_mnist
        self.dataiter_notmnist = self.data_loader.dataiter_notmnist
        self.testiter_notmnist = self.data_loader.testiter_notmnist

        self.generation = 0
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

        self.targets.append(self.labels.numpy())

    def create_individual(self):
        # get weights, biases from networks and flatten them
        # convolutional network parameters ###
        conv_ensembles = []
        with torch.no_grad():
            # weights for layers, conv1, conv2, fc1
            # conv1_weights = self.conv_net.state_dict()['conv1.weight'].view(-1).numpy()
            # conv2_weights = self.conv_net.state_dict()['conv2.weight'].view(-1).numpy()
            # fc1_weights = self.conv_net.state_dict()['fc1.weight'].view(-1).numpy()

            # bias
            # conv1_bias = self.conv_net.state_dict()['conv1.bias'].numpy()
            # conv2_bias = self.conv_net.state_dict()['conv2.bias'].numpy()
            # fc1_bias = self.conv_net.state_dict()['fc1.bias'].numpy()

            # stack everything into a vector of
            # conv1_weights, conv1_bias, conv2_weights, conv2_bias,
            # fc1_weights, fc1_bias
            # params = np.hstack((conv1_weights, conv1_bias, conv2_weights,
            #                     conv2_bias, fc1_weights, fc1_bias))
            length = 0
            for key in self.conv_net.state_dict().keys():
                length += self.conv_net.state_dict()[key].nelement()

            # conv_ensembles.append(params)
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
                conv_ensembles.append(np.random.uniform(-1, 1,
                                                        size=length))
            # convert targets to numpy()
            targets = tuple([label.numpy() for label in self.labels])
            return dict(conv_params=np.array(conv_ensembles),
                        targets=targets,
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
        conv_params = np.mean(ensembles, axis=0)
        ds = self._shape_parameter_to_conv_net(conv_params)
        self.conv_net.set_parameter(ds)
        print('---- Train -----')
        print('Generation ', self.generation)
        generation_change = 8
        dataset_change = 500
        with torch.no_grad():
            inputs = self.inputs
            labels = self.labels

            if self.generation % generation_change == 0 and self.generation < dataset_change:
                self.inputs, self.labels = self.dataiter_mnist()
                print('New MNIST set used at generation {}'.format(
                    self.generation))
                # append the outputs
                self.targets.append(self.labels.numpy())
            elif model.generation >= dataset_change and self.generation % generation_change == 0:
                if model.generation == generation_change:
                    self.test_input, self.test_label = self.testiter_notmnist()
                self.inputs, self.labels = self.dataiter_notmnist()
                print('New notMNIST set used at generation {}'.format(
                    self.generation))
                # append the outputs
                self.targets.append(self.labels.numpy())

            outputs = self.conv_net(inputs)
            self.output_activity_train.append(outputs.numpy())
            conv_loss = self.criterion(outputs, labels).item()
            train_cost = _calculate_cost(_encode_targets(labels, 10),
                                         outputs.numpy(), 'MSE')
            train_acc = score(labels.numpy(), np.argmax(outputs.numpy(), 1))
            print('Cost: ', train_cost)
            print('Accuracy: ', train_acc)
            print('Loss:', conv_loss)
            self.train_cost.append(train_cost)
            self.train_acc.append(train_acc)
            self.train_pred.append(np.argmax(outputs.numpy(), 1))

            print('---- Test -----')
            test_output = self.conv_net(self.test_input)
            test_output = test_output.numpy()
            test_acc = score(self.test_label.numpy(), np.argmax(test_output, 1))
            test_cost = _calculate_cost(_encode_targets(self.test_label, 10),
                                        test_output, 'MSE')
            print('Test accuracy', test_acc)
            self.test_acc.append(test_acc)
            self.test_pred.append(np.argmax(test_output, 1))
            self.test_cost.append(test_cost)
            self.output_activity_test.append(test_output)
            print('-----------------')
            conv_params = []
            for c in ensembles:
                ds = self._shape_parameter_to_conv_net(c)
                self.conv_net.set_parameter(ds)
                conv_params.append(self.conv_net(inputs).numpy().T)

            outs = {
                'conv_params': np.array(conv_params),
                'conv_loss': float(conv_loss),
                'input': self.inputs.squeeze().numpy(),
                'targets': self.labels.numpy()
                }
        return outs

    def _shape_parameter_to_conv_net(self, params):
        param_dict = dict()
        start = 0
        for key in self.conv_net.state_dict().keys():
            shape = self.conv_net.state_dict()[key].shape
            length = self.conv_net.state_dict()[key].nelement()
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
    root_ = '/home/yegenoglu/Documents/toolbox/L2L/l2l/optimizees/multitask'
    path_ = './dataloader.npy'
    n_ensembles = 1000
    conv_loss_mnist = []
    np.random.seed(0)
    batch_size = 32
    model = MnistOptimizee(root=root_, batch_size=batch_size, seed=0,
                           n_ensembles=n_ensembles, path=path_)
    conv_ens = None
    gamma = np.eye(10) * 0.01
    enkf = EnKF(tol=1e-5,
                maxit=1,
                stopping_crit='',
                online=False,
                shuffle=False,
                n_batches=1,
                converge=False)
    for i in range(1000):
        model.generation = i + 1
        if i == 0:
            out = model.create_individual()
            conv_ens = out['conv_params']
            out = model.set_parameters(conv_ens)
            print('loss {} generation {}'.format(out['conv_loss'],
                                                 model.generation))
        t = time.time()
        enkf.fit(data=out['input'],
                 ensemble=conv_ens,
                 ensemble_size=n_ensembles,
                 moments1=np.mean(conv_ens, axis=0),
                 observations=out['targets'],
                 model_output=out['conv_params'], noise=0.0,
                 gamma=gamma, p=None)
        print('done in {} s'.format(time.time() - t))
        conv_ens = enkf.ensemble
        out = model.set_parameters(conv_ens)
        conv_loss_mnist.append(out['conv_loss'])
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
        'test_targets': model.test_label.numpy(),
        'ensemble': conv_ens,
    }
    np.save('conv_params.npy', param_dict)
    # d = {
    #     'total_cost': {'total_cost': conv_loss_mnist}
    # }
    # plotter = Plotter(d)
    # plotter.plot()
    # plotter.get_plots()
