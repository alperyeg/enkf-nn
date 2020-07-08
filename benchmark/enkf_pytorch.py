import numpy as np
import abc
import time
import torch
from abc import ABCMeta
from torch.nn.functional import one_hot

# TODO: revise the documentation


class KalmanFilter(metaclass=ABCMeta):

    @abc.abstractmethod
    def fit(self, data, ensemble, ensemble_size, moments1,
            observations, model_output, gamma, noise):
        pass


class EnsembleKalmanFilter(KalmanFilter):
    def __init__(self, tol, maxit, stopping_crit, n_batches=32,
                 shuffle=True, online=False, converge=False,
                 loss_function='norm'):
        """
        Ensemble Kalman Filter (EnKF)

        EnKF following the formulation found in Iglesias et al. (2013), 
        The Ensemble Kalman Filter for Inverse Problems.

        :param tol: float, tolerance value for convergence
        :param maxit: int, maximum number of iterations
        :param stopping_crit: str, stopping criterion,
            `discrepancy`: checks if the actual misfit is smaller or equal
            to the noise
            `relative`: checks if the absolute value between actual and
            previous misfit is smaller than given tolerance `tol`
            otherwise calculates the loss between iteration `n` and `n-1` and
            stops if the difference is `< tol`
        :param shuffle: bool, True if the dataset should be shuffled,
            Default is `True`.
        :param n_batches, int,  number of batches to used in mini-batch. If set
            to `1` uses the whole given dataset. Default is `32`.
        :param online, bool, True if one random data point is requested,
            between [0, dims], otherwise do mini-batch, Default is False
        :param converge, bool, Checks and stops the iteration and updating step
            if convergence is reached, Default is `False`.
        :param loss_function, string, name of the loss function
           `MAE` is the Mean Absolute Error or l1 - loss
           `MSE` is the Mean Squared Error or l2 - loss
           `CE` cross-entropy loss
           `norm` norm-2 or Frobenius norm,
           Default is `norm`

        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.Cpp = None
        self.Cup = None
        self.ensemble = None
        self.observations = None
        self.data = None
        self.e = None
        self.r = None
        self.misfit = None
        self.u_exact = None
        self.norm_uexact2 = None
        self.norm_p2 = None
        self.m1 = None

        self.maxit = maxit
        self.shuffle = shuffle
        self.stopping_crit = stopping_crit
        self.tol = tol
        self.converge = converge
        self.online = online
        self.n_batches = n_batches
        self.loss_function = loss_function
        self.gamma = 0.
        self.gamma_s = 0
        self.dims = 0
        self.times = {}

    def fit(self, data, ensemble, ensemble_size, moments1, observations,
            model_output, gamma, noise=0.):
        """
        Prediction and update step of the EnKF

        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param observations: nd numpy array, observation or targets
        :param model_output: nd numpy array, output of the model
            In terms of the Kalman Filter the model maps the ensembles (dim n)
            into the observed data `y` (dim k). E.g. network output activity
        :param noise: nd numpy array, Noise can be added to the model (for `gamma`) 
            and is used in the misfit calculation for convergence.
            E.g. multivariate normal distribution. Default is `0.0`
        :param  gamma: nd numpy array, Normalizes the model-data distance in the
            update step, :`noise * I` (I is identity matrix) or
            :math:`\\gamma=I` if `noise` is zero
        :return self, Possible outputs are, if calculated:
            ensembles: nd numpy array, optimized `ensembles`
            Cpp: nd numpy array, covariance matrix of the model output
            Cup: nd numpy array, covariance matrix of the model output and the ensembles 
        """
        # get shapes
        self.gamma_s, self.dims = _get_shapes(observations, model_output)

        if isinstance(gamma, (int, float)):
            if float(gamma) == 0.:
                self.gamma = np.eye(self.gamma_s)
        else:
            self.gamma = gamma

        self.times = {'Cpp':[], 'Cup':[], 'update':[], 'lstsq': [], 'mm': []}

        # copy the data so we do not overwrite the original arguments
        self.ensemble = ensemble.clone()
        self.observations = observations.clone()
        self.observations = _encode_targets(observations, self.gamma_s)
        self.data = data.clone()
        # convert to pytorch
        self.ensemble = torch.as_tensor(
            self.ensemble, device=self.device, dtype=torch.float32)
        self.observations = torch.as_tensor(
            self.observations, device=self.device, dtype=torch.float32)
        self.data = torch.as_tensor(self.data, device=self.device)
        self.gamma = torch.as_tensor(
            self.gamma, device=self.device, dtype=torch.float32)
        model_output = torch.as_tensor(
            model_output, device=self.device, dtype=torch.float32)

        for i in range(self.maxit):
            if (i % 100) == 0:
                print('Iteration {}/{}'.format(i, self.maxit))
            if self.shuffle:
                data, observations = _shuffle(self.data, self.observations)

            # now get mini_batches
            if self.n_batches > self.dims:
                num_batches = 1
            else:
                num_batches = self.n_batches
            mini_batches = _get_batches(
                num_batches, shape=self.dims, online=self.online)
            mini_batches = torch.as_tensor(mini_batches, device=self.device)
            for idx in mini_batches:
                # in case of online learning idx should be an int
                # put it into a list to loop over it
                for d in idx:
                    # now get only the individuals output according to idx
                    g_tmp = model_output[:, :, d]
                    # Calculate the covariances
                    t = time.time()
                    Cpp = _cov_mat(g_tmp, g_tmp, ensemble_size)
                    self.times['Cpp'].append(time.time()-t)
                    t = time.time()
                    Cup = _cov_mat(self.ensemble, g_tmp, ensemble_size)
                    self.times['Cup'].append(time.time()-t)
                    t = time.time()
                    self.ensemble = _update_step(self.ensemble,
                                                 self.observations[d],
                                                 g_tmp, self.gamma, Cpp, Cup)
                    self.times['update'].append(time.time()-t)

        return self


def _update_step(ensemble, observations, g, gamma, Cpp, Cup):
    """
    Update step of the kalman filter
    Calculates the covariances and returns new ensembles
    """
    # return ensemble + (Cup @ np.linalg.lstsq(Cpp+gamma, (observations - g).T)[0]).T
    return torch.mm(Cup, torch.lstsq((observations-g).t(), Cpp+gamma)[0]).t() + ensemble
    # c = Cpp + gamma
    # o = observations-g
    # o = o.t()
    # t = time.time()
    # lstsq = torch.lstsq(o, c)[0]
    # times['lstsq'].append(time.time() - t)
    # t = time.time()
    # r = torch.mm(Cup, lstsq)
    # r = r.t()
    # times['mm'].append(time.time() - t)
    # return r + ensemble


def _cov_mat(x, y, ensemble_size):
    """
    Covariance matrix
    """
    # x_bar = _get_mean(x)
    # y_bar = _get_mean(y)

    # cov = 0.0
    return torch.tensordot((x - x.mean(0)), (y - y.mean(0)),
                           dims=([0], [0])) / ensemble_size


def _get_mean(x):
    """
    Depending on the shape returns the correct mean
    """
    if x.shape[1] == 1:
        return np.mean(x)
    return np.mean(x, axis=0)


def _get_shapes(observations, model_output):
    """
    Returns individual shapes

    :returns gamma_shape, length of the observation (here: size of last layer
                          of network)
    :returns dimensions, number of observations (and data)
    """
    if len(model_output.size()) > 2:
        gamma_shape = model_output.shape[1]
    else:
        gamma_shape = model_output.shape[0]
    dimensions = observations.shape[0]
    return gamma_shape, dimensions


def _one_hot_vector(index, shape):
    """
    Encode targets into one-hot representation
    """
    target = np.zeros(shape)
    target[index] = 1.0
    return target


def _encode_targets(targets, shape):
    #return np.array(
    #    [_one_hot_vector(targets[i], shape) for i in range(targets.shape[0])])
    return one_hot(targets, shape).float()



def _shuffle(data, targets):
    """
    Shuffles the data and targets by permuting them
    """
    indices = np.random.permutation(targets.shape[0])
    return data[indices], targets[indices]


def _get_batches(n_batches, shape, online):
    """
    :param n_batches, int, number of batches
    :param shape, int, shape of the data
    :param online, bool, True if one random data point is requested,
                         between [0, dims], otherwise do mini-batch
    """
    if online:
        return [np.random.randint(0, shape)]
    else:
        num_batches = n_batches
        mini_batches = _mini_batches(shape=shape,
                                     n_batches=num_batches)
        return mini_batches


def _mini_batches(shape, n_batches):
    """
    Splits the data set into `n_batches` of shape `shape`
    """
    return np.array_split(range(shape), n_batches)
