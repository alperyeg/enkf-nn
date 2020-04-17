import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sns.set(style="white")
sns.set_color_codes("dark")


def plot_total_cost(total_cost, title='Cost'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$c$')
    ax.semilogy(range(len(total_cost)), total_cost, '.-')
    fig.savefig(title+'.pdf')
    plt.show()


def plot_distributions(dist, **kwargs):
    fig, ax = plt.subplots()
    ax.set_title(kwargs['title'])
    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])
    # ax.set_ylim(0, 0.5)
    sns.distplot(dist, color='b', ax=ax)
    # ax.hist(dist, range=(0, 10))
    fig.savefig(kwargs['title'] + '.pdf')
    plt.show()


def plot_activity_distributions(dist1, dist2, dist3, **kwargs):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    plt.title('Distribution - Output activities')
    sns.distplot(dist1, bins=10, color='b', ax=ax1)
    sns.distplot(dist2, bins=10, color='b', ax=ax2)
    sns.distplot(dist3, bins=10, color='b', ax=ax3)
    ax1.set_title(kwargs['title1'])
    ax2.set_xlabel(kwargs['xlabel'])
    ax1.set_ylabel(kwargs['ylabel'])
    ax2.set_title(kwargs['title2'])
    ax3.set_title(kwargs['title3'])
    fig.savefig('Distribution - Output activities.pdf')
    plt.show()


def plot_accuracies(acc1, acc2, **kwargs):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=(6, 6))
    if kwargs.get('smooth', False):
        ax1.set_xticklabels(
            np.insert(np.linspace(0, 7000, 10).astype(int), 0, 0))
    ax1.plot(acc1, '.-')
    ax2.plot(acc2, '.-')
    ax1.set_ylabel(kwargs['ylabel'])
    ax1.set_xlabel(kwargs['xlabel'])
    ax1.set_title(kwargs['title1'])
    ax2.set_title(kwargs['title2'])
    plt.tight_layout()
    fig.savefig('Accuracy.pdf')
    plt.show()


def plot_accuracies_smooth(acc1, **kwargs):
    fig, ax1 = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(6, 6))
    if kwargs.get('smooth', False):
        ax1.set_xticklabels(
            np.insert(np.linspace(0, 7000, 10).astype(int), 0, 0))
    ax1.plot(acc1, '.-')
    yhat = savgol_filter(acc1, 1001, 3)
    yhat = np.insert(yhat, 0, 0)
    ax1.plot(yhat, linewidth=3)
    ax1.set_ylabel(kwargs['ylabel'])
    ax1.set_xlabel(kwargs['xlabel'])
    ax1.set_title(kwargs['title'])
    plt.tight_layout()
    fig.savefig('Accuracy_smooth.pdf')
    plt.show()


if __name__ == '__main__':
    # load parameters
    path = './conv_params_6000.npy'
    params = np.load(path, allow_pickle=True).item()
    # dictionary with parameter to the plot
    plot_dict = {'title1': 'Targets',
                 'title2': 'Train Prediction',
                 'title3': 'Test Prediction',
                 'xlabel': 'Digits', 'ylabel': 'Frequency'}
    # plot distributions
    train_act = []
    test_act = []
    train_targets = []

    for tr, tt, trt in zip(params['train_act'], params['test_act'],
                           params['train_targets']):
        train_act.extend(np.argmax(tr, 1))
        test_act.extend(np.argmax(tt, 1))
        train_targets.extend(trt)

    plot_activity_distributions(
        train_targets,
        train_act,
        test_act,
        **plot_dict)

    plot_dict = {'title': 'Accuracy',
                 'xlabel': 'Iterations', 'ylabel': 'Accuracy',
                 'title1': 'Train Accuracy', 'title2': 'Test Accuracy',
                 'smooth': False}

    plot_accuracies(
        np.array(params['train_acc']).ravel(),
        np.array(params['test_acc']).ravel(),
        **plot_dict
    )

    plot_accuracies_smooth(
        np.array(params['test_acc']).ravel(),
        **plot_dict
    )

    plot_dict = {'title': 'Distribution Mean Ensembles',
                 'xlabel': '', 'ylabel': 'Frequency'}
    plot_distributions(np.array(params['ensemble']).mean(0), **plot_dict)

    plot_total_cost(params['test_cost'], title='Test cost')
