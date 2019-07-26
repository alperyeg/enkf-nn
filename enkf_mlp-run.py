import numpy as np
import nn as netw
import torch
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

from enkf import EnsembleKalmanFilter as EnKF
from numpy.linalg import norm
from enkf import _encode_targets


def plot_total_cost(total_cost, title='Cost'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$c$')
    ax.plot(range(len(total_cost)), total_cost, '.-')
    fig.savefig(title+'.eps', format='eps')


def plot_distributions(dist, title='Distribution'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Digits')
    ax.set_ylabel('Frequency')
    ax.hist(dist, range=(0, 10))
    fig.savefig(title+'.eps', format='eps')    


def create_ensembles(n_ensembles, net):
    ensembles = []
    weight_shapes = net.get_weights_shapes()
    cumulative_num_weights_per_layer = np.cumsum(
        [np.prod(weight_shape) for weight_shape in weight_shapes])

    flattened_weights = np.empty(cumulative_num_weights_per_layer[-1])
    for i, weight_shape in enumerate(weight_shapes):
        if i == 0:
            flattened_weights[:cumulative_num_weights_per_layer[i]] = \
                np.random.randn(np.prod(weight_shape)) / np.sqrt(
                    weight_shape[1])
        else:
            flattened_weights[cumulative_num_weights_per_layer[i - 1]:
                              cumulative_num_weights_per_layer[i]] = \
                np.random.randn(np.prod(weight_shape)) / np.sqrt(
                    weight_shape[1])

    ensembles.append(flattened_weights)
    for _ in range(n_ensembles - 1):
        ensembles.append(np.random.uniform(-1, 1, len(flattened_weights)))
    return np.array(ensembles)


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
                                                    num_workers=0)

    testset_mnist = torchvision.datasets.MNIST(root=root,
                                               train=False,
                                               download=True,
                                               transform=transform)
    testload_mnist = torch.utils.data.DataLoader(testset_mnist,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
    return trainloader_mnist, testload_mnist


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
    n_correct = np.count_nonzero(y == x)
    n_total = len(y)
    sc = n_correct / n_total
    return sc


def get_data(description='exact', mnist_digits=None):
    if description == 'exact':
        outs = np.load('net_outs.npy').item()
        weights = outs['weights']
        observations = outs['output']
        return weights, observations
    elif description.upper() == 'MNIST':
        observations = mnist_digits.target
        return None, observations


def main():
    np.random.seed(0)
    n_ensemble = 1000
    net = netw.NeuralNetworkClassifier(n_input=784, n_hidden=60, n_output=10)
    root = '/home/yegenoglu/Documents/toolbox/L2L/l2l/optimizees/multitask'
    batch_size = 40
    trainloader, testloader = load_data(root, batch_size)
    dataiter = iter(trainloader)
    testiter = iter(testloader)
    test_images, test_labels = testiter.next()
    test_images = test_images.squeeze().numpy().reshape(batch_size, -1)
    test_labels = test_labels.numpy()
    ensembles = create_ensembles(n_ensemble, net) # sample_ensembles(n_ensemble, weights)
    gamma = np.eye(10) * 0.01
    enkf = EnKF(tol=1e-5,
                maxit=1,
                stopping_crit='',
                online=False,
                shuffle=False,
                n_batches=1,
                converge=False)
    cost_train_list = []
    cost_test_list = []
    score_train_list = []
    score_test_list = []
    labels_list = []
    output_train_list = []
    output_test_list = []
    activation_train_list = []
    for i in range(50):
        data_images, observations = dataiter.next()
        data_images = data_images.squeeze().numpy().reshape(batch_size, -1)
        observations = observations.numpy()
        model_outputs = [
            net.get_output_activation(data_images,
                                      *net.flatten_to_net_weights(e))
            for e in ensembles]
        model_outputs = np.array(model_outputs)
        t = time.time()
        enkf.fit(data=data_images,
                 ensemble=ensembles,
                 ensemble_size=n_ensemble,
                 moments1=np.mean(ensembles, axis=0),
                 u_exact=None,
                 observations=observations,
                 model_output=model_outputs, noise=0.0,
                 gamma=gamma, p=None)
        ensembles = enkf.ensemble
        print('done in {} s'.format(time.time() - t))
        weights = np.mean(ensembles, 0)
        net.set_weights(*net.flatten_to_net_weights(weights))
        output = net.get_output_activation(data_images,
                                           *net.flatten_to_net_weights(
                                               weights))
        output_test = net.get_output_activation(test_images,
                                                *net.flatten_to_net_weights(
                                                    weights))
        cost = _calculate_cost(_encode_targets(observations, 10),
                               _encode_targets(np.argmax(output, 0), 10),
                               'MSE')
        scores = score(observations, np.argmax(output, 0)) * 100

        cost_test = _calculate_cost(y=test_labels, y_hat=np.argmax(output_test, 0),
                                    loss_function='MSE')
        score_test = score(test_labels, np.argmax(output_test, 0)) * 100
        print('---- Train -----')
        print('targets: ', observations)
        print('predict: ', np.argmax(output, 0))
        print('score: ', scores)
        print('cost {} in iteration {}/{}'.format(cost, i+1, 100))
        print('---- Test -----')
        print('accuracy: ', score_test)
        print('-----------------')
        cost_train_list.append(cost)
        cost_test_list.append(cost_test)
        score_train_list.append(scores)
        score_test_list.append(score_test)
        labels_list.extend(observations)
        output_train_list.extend(np.argmax(output, 0))
        output_test_list.extend(np.argmax(output_test, 0))
        activation_train_list.append(output)

    plot_total_cost(cost_train_list, title='Training_loss')
    plot_total_cost(cost_test_list, title='Test_loss')
    plot_total_cost(score_train_list, title='Training_accuracy')
    plot_total_cost(score_test_list, title='Test_accuracy')
    plot_distributions(labels_list, 'Targets')
    plot_distributions(output_train_list, 'Train_prediction')
    plot_distributions(output_test_list, 'Test_prediction')
    params = {
        'cost_train_list': cost_train_list,
        'cost_test_list': cost_test_list,
        'score_train_list': score_train_list,
        'score_test_list': score_test_list,
        'labels_list': labels_list,
        'output_train_list': output_train_list,
        'output_test_list': output_test_list,
        'activation_train_list': activation_train_list,
        'ensembles': ensembles
    }
    np.save('params.npy', params)


if __name__ == '__main__':
    main()
