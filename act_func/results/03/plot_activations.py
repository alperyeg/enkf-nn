import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

sns.set(style="white")
sns.set_color_codes("dark")


def weight_distributions_per_layer(iterations,
                                   suptitle='Ensembles - Weights',
                                   bins=None,
                                   savepath='',
                                   rand=False):
    fig, axes = plt.subplots(1, len(iterations), sharey=True)
    # fig.tight_layout()
    if rand:
        rnd = np.random.randint(0, len(iterations[0][1]['ensemble']))
    for i, params in enumerate(iterations):
        if rand:
            dist = np.array(params[1]['ensemble'])[rnd],
        else:
            dist = np.array(params[1]['ensemble']).mean(0),
        ax = axes[i]
        sns.distplot(dist, bins=bins, color='b', ax=ax)
        ax.set_title('{}'.format(params[0]).strip('conv_params .npy'))
    if rand:
        fig.suptitle(suptitle + '\n' + 'Ensemble {}'.format(rnd))
    else:
        fig.suptitle(suptitle)
    plt.savefig(savepath)
    plt.show()


def activation_functions_dist_iteration(act_func, savepath=''):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')
    act1_mean = act_func['act1_mean']
    act1_std = act_func['act1_std']
    sns.distplot(act1_mean, label='act1_mean', ax=ax1)
    sns.distplot(act1_std, label='act1_std', ax=ax1)
    act2_mean = act_func['act2_mean']
    act2_std = act_func['act2_std']
    sns.distplot(act2_mean, label='act2_mean', ax=ax2)
    sns.distplot(act2_std, label='act2_std', ax=ax2)
    act3_mean = act_func['act3_mean']
    act3_std = act_func['act3_std']
    sns.distplot(act3_mean, label='act3_mean', ax=ax3)
    sns.distplot(act3_std, label='act3_std', ax=ax3)
    plt.legend()
    plt.show()


def activations_per_layer(iteration):
    pass


def activations_mean_std_error(act_func, errorevery=10, savepath=''):
    act1_mean = act_func['act1_mean'][::8]
    act1_std = act_func['act1_std'][::8]
    act2_mean = act_func['act2_mean'][::8]
    act2_std = act_func['act2_std'][::8]
    plt.errorbar(range(len(act1_mean)), act1_mean, act1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(act2_mean)), act2_mean, act2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.legend()
    plt.savefig('act_func_mean_std.eps')
    plt.show()


if __name__ == '__main__':
    path = "./"
    conv_params = {}
    files = []
    for file in os.listdir(path):
        if file.endswith('2000.pt'):
            conv_params[file] = torch.load(file, map_location='cpu')
            files.append(file)
    # it = ['conv_params_0.npy', 'conv_params_500.npy', 'conv_params_3000.npy']
    # iters = [(f, conv_params[f]) for f in sorted(conv_params.keys())]
    # iters = [(f, conv_params[f]) for f in it]
    # weight_distributions_per_layer(iters, savepath='dist_ensembles_iterations0_rand.pdf', rand=False)
    activation_functions_dist_iteration(
        conv_params['conv_params_2000.pt']['act_func'])
    activations_mean_std_error(conv_params['conv_params_2000.pt']['act_func'])
