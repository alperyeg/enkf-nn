import numpy as np
import abc
import torch
from numpy.linalg import norm, inv, solve
from abc import ABCMeta

# TODO: revise the documentation


class KalmanFilter(metaclass=ABCMeta):

    @abc.abstractmethod
    def fit(self, data, ensemble, ensemble_size, moments1,
            observations, model_output, gamma, noise, p, u_exact):
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
        self.M = []
        self.E = []
        self.R = []
        self.AE = []
        self.AR = []
        self.total_cost = []

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

    def fit(self, data, ensemble, ensemble_size, moments1,
            observations, model_output, gamma, noise=0., p=None, u_exact=None):
        """
        Prediction and update step of the EnKF

        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param moments1: nd numpy array, first moment (mean)
        :param u_exact: nd numpy array, exact control, e.g. if weight sof the model
                   are known. Default is `None`
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
        :param p: nd numpy array
            Exact solution given by :math:`G * u_exact`, where `G` is inverse
            of a linear elliptic function `L`, it maps the control into the
            observed data, Default is `None`
        :return self, Possible outputs are, if calculated:
            ensembles: nd numpy array, optimized `ensembles`
            Cpp: nd numpy array, covariance matrix of the model output
            Cup: nd numpy array, covariance matrix of the model output and the 
                ensembles 
            M: nd numpy array, Misfit
                Measures the quality of the solution at each iteration
            E: nd numpy array, Deviation of :math:`v^j` from the mean,
                where the :math:`v^j` is the j-th sample of the approximated
                distribution of samples
            R: nd numpy array, Deviation of :math:`v^j` from the "true" solution
                :math:`u^{\\dagger}`, see (Herty2018 eq.29)
            AE: nd numpy array, Deviation of :math:`v^j` under the application of
                the model `G`, see (Herty2018 eq.28)
            AR: nd numpy array, Deviation of from the "true" solution under
                the application of model `G`, see (Herty2018 eq.28)
           total_cost: list, contains the costs as defined in `loss_function`
        """
        # get shapes
        self.gamma_s, self.dims = _get_shapes(observations, model_output)

        if isinstance(gamma, (int, float)):
            if float(gamma) == 0.:
                self.gamma = np.eye(self.gamma_s)
        else:
            self.gamma = gamma

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
                    Cpp = _cov_mat(g_tmp, g_tmp, ensemble_size)
                    Cup = _cov_mat(self.ensemble, g_tmp, ensemble_size)
                    self.ensemble = _update_step(self.ensemble,
                                                 self.observations[d],
                                                 g_tmp, self.gamma, Cpp, Cup)

            # m = torch.distributions.Normal(self.ensemble.mean(),
            #                                self.ensemble.std())
            # self.ensemble += m.sample(self.ensemble.shape)
            cov = 0.01 * _cov_mat(self.ensemble, self.ensemble, ensemble_size)
            rnd = torch.randn(size=(self.ensemble.shape[1], ensemble_size), device=self.device)
            mm = torch.mm(cov, rnd).to(self.device)
            self.ensemble += torch.zeros(self.ensemble.shape[1]).to(self.device) + mm.t()
        return self


# @jit(nopython=True, parallel=True)
def _update_step(ensemble, observations, g, gamma, Cpp, Cup):
    """
    Update step of the kalman filter
    Calculates the covariances and returns new ensembles
    """
    # return ensemble + (Cup @ np.linalg.lstsq(Cpp+gamma, (observations - g).T)[0]).T
    return torch.mm(Cup, torch.lstsq((observations-g).t(), Cpp+gamma)[0]).t() + ensemble


# @jit(parallel=True)
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


def _calculate_cost(y, y_hat, loss_function='BCE'):
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


def _l1_regularization(weights, lambda_=0.1):
    """
    Compute L1-regularization cost.
    """
    return (lambda_ / 2.0) * np.sum(np.abs(weights))


def _l2_regularization(weights, lambda_=0.1):
    """
    Compute L2-regularization cost.
    """
    return lambda_ * np.sum(weights ** 2)


def _calculate_misfit(ensemble, ensemble_size, misfit, r, g_all, u_exact,
                      noise):
    """
    Calculates and returns the misfit and the deviation from the true solution
    """
    for d in range(ensemble_size):
        r[d] = ensemble[d] - u_exact
        misfit[d] = g_all[d] * r[d, 0] - noise
    return misfit, r


def _one_hot_vector(index, shape):
    """
    Encode targets into one-hot representation
    """
    target = np.zeros(shape)
    target[index] = 1.0
    return target


def _encode_targets(targets, shape):
    return np.array(
        [_one_hot_vector(targets[i], shape) for i in range(targets.shape[0])])


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
