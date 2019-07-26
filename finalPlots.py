import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, plot_dict):
        """
        A class to generate plots, based on the factory design

        :param plot_dict: dict, Keywords define which plots should be
        calculated. Supported are `deviation`, `reconstruction`, `misfit`,
        `residuals`, `control`, `total_cost`
        """
        self.d = plot_dict
        self.plot_names = ['deviation', 'reconstruction', 'misfit',
                           'residuals', 'control', 'total_cost']
        self.all_plots = {}

    def plot(self, show=True, save=False):
        """
        Plotting functionality

        :param show: bool, if the plots should be shown, Default is `True`
        :param save: bool, if the plots should be saved, Default is `False`
        """
        if not self.all_plots:
            self.get_plots()
        for k, v in self.all_plots.items():
            if show:
                plt.show()
            if save:
                v[0].savefig(k+'.eps', format='eps')

    def _get_plot(self, key, v):
        if key == 'deviation':
            return self.plot_deviation(**v)
        if key == 'reconstruction':
            return self.plot_reconstruction(**v)
        if key == 'misfit':
            return self.plot_misfit(**v)
        if key == 'residuals':
            return self.plot_residuals(**v)
        if key == 'control':
            return self.plot_control(**v)
        if key == 'total_cost':
            return self.plot_total_cost(**v)

    def get_plots(self):
        """
        Returns all plots

        Return `fig` and `axes` from a `matplotlib.pyplot.subplots`
        :return:
            figure: Figure part of the plot
            axes: One or more axes of the plot, depending on the plot
        """
        if not self.all_plots:
            for k, v in self.d.items():
                if k in self.plot_names:
                    self.all_plots[k] = self._get_plot(k, v)
                else:
                    raise ValueError("{} - not supported".format(k))
        return self.all_plots

    @staticmethod
    def plot_deviation(E):
        fig, ax = plt.subplots()
        ax.set_title('Deviation from the mean')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('v')
        ax.semilogy(range(len(E)), E, '.-')
        return fig, ax

    @staticmethod
    def plot_reconstruction(x, G, m1, p):
        fig, ax = plt.subplots()
        ax.set_title(r'Exact solution with $u(x)=1$ vs Reconstruction')
        ax.set_xlabel('x')
        ax.plot(x, G @ m1, 'x-')
        ax.plot(x, p, 'k-')
        return fig, ax

    @staticmethod
    def plot_misfit(M):
        fig, ax = plt.subplots()
        ax.set_title('Misfit')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\vartheta$')
        ax.semilogy(range(len(M)), M, '.-')
        return fig, ax

    @staticmethod
    def plot_residuals(R):
        fig, ax1 = plt.subplots()
        ax1.set_title('Residuals')
        ax1.set_ylabel('r')
        ax1.semilogy(range(len(R)), R, '.-')
        return fig, ax1

    @staticmethod
    def plot_control(x, m1, u):
        fig, ax = plt.subplots()
        ax.set_title('Reconstruction of the control')
        ax.set_xlabel('x')
        ax.plot(x, m1)
        ax.plot(x, u(x), 'k-')
        return fig, ax

    @staticmethod
    def plot_total_cost(total_cost):
        fig, ax = plt.subplots()
        ax.set_title('Total Cost')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$c$')
        ax.semilogy(range(len(total_cost)), total_cost, '.-')
        return fig, ax

