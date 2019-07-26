import numpy as np
import nn

from sklearn.datasets import load_digits
# from updateEnKF import update_enknf
from enkf import EnsembleKalmanFilter as EnKF
from finalPlots import Plotter


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
        ensembles.append(np.random.randn(len(flattened_weights)))
    return np.array(ensembles)


def sample_ensembles(n_ensembles, weights_):
    ensembles = []
    for _ in range(n_ensembles):
        # sample a small value and add it to the weights
        sample = np.random.uniform(-1e-5, 1e-5)
        ensembles.append(
            weights_ + sample)
    return np.array(ensembles)


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
    mnist_digits = load_digits()
    n_images = len(mnist_digits.images)  # 1797
    n_input = np.prod(mnist_digits.images.shape[1:])
    data_images = mnist_digits.images.reshape(
        n_images, -1) / 16.  # -> 1797 x 64
    weights, observations = get_data("exact", mnist_digits)
    n_ensemble = 500
    net = nn.NeuralNetworkClassifier(n_input=n_input, n_hidden=10, n_output=10)
    ensembles = sample_ensembles(n_ensemble, weights)
    model_outputs = [
        net.get_output_activation(data_images, *net.flatten_to_net_weights(e))
        for e in ensembles]
    model_outputs = np.array(model_outputs)
    enkf = EnKF(tol=1e-5,
                maxit=3000,
                stopping_crit='',
                online=True,
                shuffle=False,
                n_batches=32,
                converge=True)
    enkf.fit(data=data_images,
             ensemble=ensembles,
             ensemble_size=n_ensemble,
             moments1=np.mean(ensembles, axis=0),
             u_exact=weights,
             model=net,
             observations=observations.T,
             model_output=model_outputs, noise=0.0,
             gamma=0., p=None)
    M = enkf.M
    R = enkf.R
    tc = enkf.total_cost
    d = {
        'misfit': {"M": M[1:]},
        'residuals': {'R': R[1:]},
        'total_cost': {'total_cost': tc}
    }
    print("R:", R)
    print("M", M)
    print("TC:", tc)
    weights = np.mean(enkf.ensemble, 0)
    net.set_weights(*net.flatten_to_net_weights(weights))
    # output = net.get_output_activation(data_images, *net.flatten_to_net_weights(weights))
    # print(net.score(data_images, observations))
    plotter = Plotter(d)
    plotter.plot()
    plotter.get_plots()


if __name__ == '__main__':
    main()
