import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import time
# import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
from conv_net import ConvNet
from enkf_pytorch import EnsembleKalmanFilter as EnKF
from enkf_pytorch import _encode_targets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    """
    Convenience class.


    """

    def __init__(self):
        self.data_mnist = None
        self.test_mnist = None
        self.data_mnist_loader = None
        self.test_mnist_loader = None

    def init_iterators(self, root, batch_size):
        self.data_mnist_loader, self.test_mnist_loader = self.load_data(
            root, batch_size)
        # here are the expensive operations
        self.data_mnist = self._make_iterator(self.data_mnist_loader)
        self.test_mnist = self._make_iterator(self.test_mnist_loader)

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
             transforms.Normalize([0.], [1.])
             ])
        # load now MNIST dataset
        trainset_mnist = torchvision.datasets.MNIST(root=root,
                                                    train=True,
                                                    download=True,
                                                    transform=transform)
        trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        num_workers=2)

        testset_mnist = torchvision.datasets.MNIST(root=root,
                                                   train=False,
                                                   download=True,
                                                   transform=transform)
        testload_mnist = torch.utils.data.DataLoader(testset_mnist,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=2)
        return trainloader_mnist, testload_mnist

    def dataiter_mnist(self):
        """ MNIST training set list iterator """
        return self.data_mnist.next()

    def testiter_mnist(self):
        """ MNIST test set list iterator """
        return self.test_mnist.next()


class MnistOptimizee(torch.nn.Module):
    def __init__(self, seed, n_ensembles, batch_size, root):
        super(MnistOptimizee, self).__init__()

        self.random_state = np.random.RandomState(seed=seed)

        self.n_ensembles = n_ensembles
        self.batch_size = batch_size
        self.root = root

        self.conv_net = ConvNet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = DataLoader()
        self.data_loader.init_iterators(self.root, self.batch_size)
        self.dataiter_mnist = self.data_loader.dataiter_mnist
        self.testiter_mnist = self.data_loader.testiter_mnist

        self.generation = 0
        self.inputs, self.labels = self.dataiter_mnist()
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        self.test_input, self.test_label = self.testiter_mnist()
        self.test_input = self.test_input.to(device)
        self.test_label = self.test_label.to(device)

        self.timings = {'shape_parameters': [], 'set_parameters': [],
                        'set_parameters_cnn': [], 'shape_parameters_ens': []}
        
        self.length = 0
        for key in self.conv_net.state_dict().keys():
            self.length += self.conv_net.state_dict()[key].nelement()


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
            # length = 0
            # for key in self.conv_net.state_dict().keys():
            #     length += self.conv_net.state_dict()[key].nelement()

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
                conv_ensembles.append(np.random.normal(0, 0.1,
                                                       size=self.length))
            return dict(conv_params=torch.as_tensor(conv_ensembles,
                                                    device=device),
                        targets=self.labels,
                        input=self.inputs.squeeze())

    def load_model(self, path='conv_params.npy'):
        print('Loading model from path: {}'.format(path))
        conv_params = np.load(path).item()
        conv_ensembles = conv_params.get('ensemble')
        return dict(conv_params=torch.as_tensor(conv_ensembles,
                                                device=device),
                    targets=self.labels,
                    input=self.inputs.squeeze())

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
        conv_params = ensembles.mean(0)
        t = time.time()
        ds = self._shape_parameter_to_conv_net(conv_params)
        self.timings['shape_parameters_ens'].append(time.time()-t)
        t = time.time()
        self.conv_net.set_parameter(ds)
        self.timings['set_parameters_cnn'].append(time.time()-t)
        print('---- Train -----')
        print('Generation ', self.generation)
        generation_change = 1
        with torch.no_grad():
            inputs = self.inputs.to(device)
            labels = self.labels.to(device)

            if self.generation % generation_change == 0:
                self.inputs, self.labels = self.dataiter_mnist()
                self.inputs = self.inputs.to(device)
                self.labels = self.labels.to(device)
                print('New MNIST set used at generation {}'.format(
                    self.generation))

            outputs, act1, act2 = self.conv_net(inputs)
            conv_loss = self.criterion(outputs, labels).item()
            train_cost = _calculate_cost(_encode_targets(labels, 10),
                                         F.softmax(outputs, dim=1), 'MSE')
            train_acc = score(labels,
                              torch.argmax(F.softmax(outputs, dim=1), 1))
            print('Cost: ', train_cost)
            print('Accuracy: ', train_acc)
            print('Loss:', conv_loss)

            print('---- Test -----')
            test_output, act1, act2 = self.conv_net(self.test_input)
            test_loss = self.criterion(test_output, self.test_label).item()
            test_acc = score(self.test_label,
                             torch.argmax(test_output, 1))
            test_cost = _calculate_cost(_encode_targets(self.test_label, 10),
                                        test_output, 'MSE')
            print('Test accuracy', test_acc)
            print('Test loss: {}'.format(test_loss))
            print('-----------------')
            conv_params = []
            for idx, c in enumerate(ensembles):
                t = time.time()
                ds = self._shape_parameter_to_conv_net(c)
                self.timings['shape_parameters'].append(time.time()-t)
                t = time.time()            
                self.conv_net.set_parameter(ds)
                self.timings['set_parameters'].append(time.time()-t)
                params, _, _ = self.conv_net(inputs)
                conv_params.append(params.t())
            conv_params = torch.stack(conv_params)
            outs = {
                'conv_params': conv_params,
                'conv_loss': float(conv_loss),
                'input': self.inputs.squeeze(),
                'targets': self.labels
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
        term1 = -y * torch.log(y_hat)
        term2 = (1 - y) * torch.log(1 - y_hat)
        return torch.sum(term1 - term2)
    elif loss_function == 'MAE':
        return torch.sum(np.absolute(y_hat - y)) / len(y)
    elif loss_function == 'MSE':
        return torch.sum((y_hat - y) ** 2) / len(y)
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
    n_correct = (y == x).sum().item()
    n_total = float(len(y))
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


def test(net, iteration, test_loader_mnist, criterion):
    net.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader_mnist):
            img = img.to(device)
            target = target.to(device)
            output, _, _ = net(img)
            loss = criterion(output, target)
            test_loss += loss.item()
            # network prediction
            pred = output.argmax(1, keepdim=True)
            # how many image are correct classified, compare with targets
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

            if idx % 10 == 0:
                print('Test Loss {} in iteration {}, idx {}'.format(
                    loss.item(), iteration, idx))
        ta = 100 * test_accuracy / len(test_loader_mnist.dataset)
        tl = test_loss / len(test_loader_mnist.dataset)
        print('------ Evaluation -----')
        print('Test accuracy: {} Average test loss: {}'.format(ta, tl))
        print('-----------------------')
    return ta, tl


def dict_values_to_numpy(dct):
    d = {}
    for k, v in dct.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cpu().numpy()
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    timings = {}
    root = '../multitask'
    n_ensembles = 2
    conv_loss_mnist = []
    # average test losses
    test_losses = []
    np.random.seed(0)
    torch.manual_seed(0)
    batch_size = 64
    model = MnistOptimizee(root=root, batch_size=batch_size, seed=0,
                           n_ensembles=n_ensembles).to(device)
    conv_ens = None
    gamma = np.eye(10) * 0.01
    enkf = EnKF(tol=1e-5,
                maxit=1,
                stopping_crit='',
                online=False,
                shuffle=False,
                n_batches=1,
                converge=False)
    rng = int(60000 / batch_size * 8)
    for i in range(1):
        model.generation = i + 1
        if i == 0:
            try:
                out = model.load_model('')
                # replace cov matrix with cov from weights (ensembles)
                # m = torch.distributions.Normal(out['conv_params'].mean(),
                #                                out['conv_params'].std())
                # model.cov = m.sample((n_ensembles, model.length))
            except FileNotFoundError as fe:
                print(fe)
                print('Model not found! Initalizaing new ensembles.')
                out = model.create_individual()
            conv_ens = out['conv_params']  # + torch.as_tensor(model.cov)
            t = time.time()
            out = model.set_parameters(conv_ens)
            model.timings['set_parameters_cnn'].append(time.time()-t)

        t = time.time()
        enkf.fit(data=out['input'],
                 ensemble=conv_ens,
                 ensemble_size=n_ensembles,
                 moments1=conv_ens.mean(0),
                 observations=out['targets'],
                 model_output=out['conv_params'], noise=0.0,
                 gamma=gamma)
        timings[str(i)] = {**model.timings, **enkf.times}
        print('done in {} s'.format(time.time() - t))
        conv_ens = enkf.ensemble
        t = time.time()
        model.set_parameters(conv_ens)
        print('model set params time ', time.time() - t)
        model.timings['set_parameters_cnn'].append(time.time()-t)
        
    last_key = np.sort([int(j) for j in list(timings)])[-1] + 1
    timings['{}'.format(last_key)] = {**model.timings, **enkf.times}
    # torch.save(timings, 'timings_ep{}.pt'.format(last_key))
