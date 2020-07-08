from enkf_pytorch import EnsembleKalmanFilter as EnKF
import pandas as pd
import numpy as np


class EnKFRunner:
    def __init__(self, data, targets, connections, model_out, n_ensembles,
                 rng):
        self.data = data
        self.targets = targets
        self.connections = connections
        self.model_out = model_out
        self.n_ensembles = n_ensembles
        self.rng = rng
        self.enkf = EnKF(maxit=1, online=False, n_batches=1)
        self.connections = self._shape_connections(connections)

    @staticmethod
    def _shape_connections(connections):
        df = pd.read_pickle(connections)
        weights = df['weight'].values
        return weights

    def run(self):
        gamma = np.eye(10) * 0.01
        for i in range(self.rng):
            self.enkf.fit(data=self.data,
                          ensemble=self.connections,
                          ensemble_size=self.n_ensembles,
                          moments1=self.connections.mean(0),
                          observations=self.targets,
                          model_output=self.model_out,
                          noise=0.0,
                          gamma=gamma,
                          )
        ens = self.enkf.ensemble
        return ens
