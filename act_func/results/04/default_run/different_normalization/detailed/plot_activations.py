import collections
import matplotlib.pyplot as plt
import numpy as np 
import os
import seaborn as sns
import torch

sns.set(style="white")
sns.set_color_codes("dark")
sns.set_context("paper")


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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')
    for k,v in act_func.items():
        act1 = v['act_func']['act1']
        act2 = v['act_func']['act2']
        act3 = v['act_func']['act3']
        sns.distplot(act1.mean(0).ravel(), ax=ax1, label='iteration {}'.format(k.strip('conv_params_ .pt')))
        sns.distplot(act2.mean(0).ravel(), ax=ax2, label='iteration {}'.format(k.strip('conv_params_ .pt')))
        sns.distplot(act3.mean(0).ravel(), ax=ax3, label='iteration {}'.format(k.strip('conv_params_ .pt')))
    ax1.set_ylim(top=30)
    ax2.set_ylim(top=30)
    ax3.set_ylim(top=30)
    plt.xlabel('Activation value')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def activations_per_layer(iteration):
    pass


def activations_mean_std_error(act_func, errorevery=10, savepath=''):
    act1_mean = act_func['act1_mean'][::8]
    act1_std = act_func['act1_std'][::8]
    act2_mean = act_func['act2_mean'][::8]
    act2_std = act_func['act2_std'][::8]
    act3_mean = act_func['act3_mean'][::8]
    act3_std = act_func['act3_std'][::8]

    plt.errorbar(range(len(act1_mean)), act1_mean, act1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(act2_mean)), act2_mean, act2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(act3_mean)), act3_mean, act3_std,
                 errorevery=errorevery, alpha=0.5, label='layer 3')
    plt.xlabel('Iteration of mini-batches')
    plt.ylabel('Activation value')
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def load_iter_act(path, startswith):
    d = collections.OrderedDict()
    files = []
    for file in os.listdir(path):
        if file.startswith(startswith):
            files.append(file)

    files = sorted(files, key=lambda x: int(x.strip('conv_params .pt')))
    print(files)
    for file in files:
        d[file] = torch.load(file)
    return d


if __name__ == '__main__':
    # path = "./"
    # conv_params = {}
    # files = []
    # for file in os.listdir(path):
    #     if file.endswith('params.pt'):
    #         conv_params[file] = torch.load(file, map_location='cpu')
    #         files.append(file)
    # it = ['conv_params_0.npy', 'conv_params_500.npy', 'conv_params_3000.npy']
    # iters = [(f, conv_params[f]) for f in sorted(conv_params.keys())]
    # iters = [(f, conv_params[f]) for f in it]
    # weight_distributions_per_layer(iters, savepath='dist_ensembles_iterations0_rand.pdf', rand=False)
    activations = load_iter_act('.', startswith='conv_params_')
    activation_functions_dist_iteration(activations, savepath='enkf_act_func_dist.pdf')
    # activations_mean_std_error(conv_params['conv_params.pt']['act_func'], savepath='enkf_act_func_mean_std.pdf')
