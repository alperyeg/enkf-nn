import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

sns.set(style="white")
sns.set_color_codes("dark")
sns.set_context("paper", font_scale=1.5, rc={
                "lines.linewidth": 2., "grid.linewidth": 0.1})


def weight_distributions_per_layer(model, model_iteration,
                                   layer='conv1.weight',
                                   suptitle='Layer 1 Weights',
                                   bins=None,
                                   savepath=''):
    fig, axes = plt.subplots(1, len(model_iteration), sharey=True)
    # fig.tight_layout()
    # batch size
    bs = 64
    for i, m in enumerate(model):
        weights = m.get(layer)
        ax = axes[i]
        dist = weights.numpy().mean(0).ravel()
        sns.distplot(dist, bins=bins, color='b', ax=ax)
        ax.set_title('{}'.format(model_iteration[i] * bs))
    fig.suptitle(suptitle)
    plt.savefig(savepath)
    plt.show()


def activation_functions_dist_iteration(act_func, savepath=''):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')

    act1 = act_func.get('act1')
    act2 = act_func.get('act2')
    act3 = act_func.get('act3')
    [sns.distplot(a.mean(0).ravel(), ax=ax1,
                  label='iteration {}'.format(i * 200))
     for i, a in enumerate(act1)]
    [sns.distplot(a.mean(0).ravel(), ax=ax2,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(act2)]
    [sns.distplot(a.mean(0).ravel(), ax=ax3,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(act3)]
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def activation_functions_dist_layer(act_func, iteration=-1, savepath=''):
    act1 = act_func.get('act1')
    act2 = act_func.get('act2')
    act3 = act_func.get('act3')
    sns.distplot(act1[iteration][0].ravel(), label='Layer 1')
    sns.distplot(act2[iteration][0].ravel(), label='Layer 2')
    sns.distplot(act3[iteration][0].ravel(), label='Layer 3')
    plt.title('Activation Value at Iteration {}'.format(iteration))
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def activation_functions_mean_std(act_func, errorevery=10, savepath=''):
    act1_mean = np.array(act_func.get('act1_mean'))
    act2_mean = np.array(act_func.get('act2_mean'))
    act3_mean = np.array(act_func.get('act3_mean'))
    act1_std = np.array(act_func.get('act1_std'))
    act2_std = np.array(act_func.get('act2_std'))
    act3_std = np.array(act_func.get('act3_std'))

    plt.errorbar(range(len(act1_mean)), act1_mean, act1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(act2_mean)), act2_mean, act2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(act3_mean)), act3_mean, act3_std,
                 errorevery=errorevery, alpha=0.4, label='layer 3')
    plt.xlabel('Iterations of mini-batches')
    plt.ylabel('Activation value')
    # plt.plot(range(len(act1_mean)), act1_mean)
    # plt.fill_between(range(len(act1_mean)), act1_mean,
    #                  act1_mean+act1_std, alpha=0.5, facecolor='blue')
    # plt.fill_between(range(len(act1_mean)), act1_mean,
    #                  act1_mean-act1_std, alpha=0.5, facecolor='blue')
    # plt.plot(range(len(act2_mean)), act2_mean)
    # plt.fill_between(range(len(act2_mean)), act2_mean,
    #                  act2_mean+act2_std, alpha=0.5, facecolor='yellow')
    # plt.fill_between(range(len(act2_mean)), act2_mean,
    #                  act2_mean-act2_std, alpha=0.5, facecolor='yellow')
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def gradients_dist_layer(grads, savepath=''):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax3.set_title('Layer 3')
    ax2.set_xlabel('Backpropagated Gradients')
    ax1.set_ylabel('Counts')

    grad1 = grads.get('conv1_grad')
    grad2 = grads.get('conv2_grad')
    grad3 = grads.get('fc1_grad')
    [sns.distplot(a.mean(0).ravel(), ax=ax1,
                  label='iteration {}'.format(i * 200))
     for i, a in enumerate(grad1)]
    [sns.distplot(a.mean(0).ravel(), ax=ax2,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(grad2)]
    [sns.distplot(a.mean(0).ravel(), ax=ax3,
                  label='iteration {}'.format(i * 200)) for i, a in
     enumerate(grad3)]
    ax2.set_ylim(0, 1000)
    ax3.set_ylim(0, 0.5e9)
    plt.legend()
    plt.show()
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0.1)


def gradients_mean_std(grads, errorevery=10, savepath=''):
    grad1_mean = grads['conv1_grad_mean']
    grad2_mean = grads['conv2_grad_mean']
    grad3_mean = grads['fc1_grad_mean']

    grad1_std = grads['conv1_grad_std']
    grad2_std = grads['conv2_grad_std']
    grad3_std = grads['fc1_grad_std']

    plt.errorbar(range(len(grad1_mean)), grad1_mean, grad1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(grad2_mean)), grad2_mean, grad2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(grad3_mean)), grad3_mean, grad3_std,
                 errorevery=errorevery, alpha=0.5, label='layer 3')
    plt.legend()
    plt.savefig(savepath)
    plt.show()


def gradients_per_epoch(grads, errorevery=10, savepath=''):
    grad1_mean = []
    grad2_mean = []
    grad3_mean = []
    grad1_std = []
    grad2_std = []
    grad3_std = []
    for k, v in grads.items():
        grad1_mean += v['conv1_grad_mean']
        grad2_mean += v['conv2_grad_mean']
        grad3_mean += v['fc1_grad_mean']
        grad1_std += v['conv1_grad_std']
        grad2_std += v['conv2_grad_std']
        grad3_std += v['fc1_grad_std']
    plt.errorbar(range(len(grad1_mean)), grad1_mean, grad1_std,
                 errorevery=errorevery, alpha=0.8, label='layer 1')
    plt.errorbar(range(len(grad2_mean)), grad2_mean, grad2_std,
                 errorevery=errorevery, alpha=0.5, label='layer 2')
    plt.errorbar(range(len(grad3_mean)), grad3_mean, grad3_std,
                 errorevery=errorevery, alpha=0.5, label='layer 3')
    # plt.xticks(range(0, 5000, 1000), range(0, 50, 10))
    # plt.plot(grad1_mean, '.', label='layer 1')
    # plt.plot(grad2_mean, '.', label='layer 2')
    # plt.plot(grad3_mean, '.', label='layer 3')
    # plt.plot(grad1_std, '*', label='layer 1 std')
    # plt.plot(grad2_std, '*', label='layer 2 std')
    # plt.plot(grad3_std, '*', label='layer 3 std')
    plt.xlabel('Epochs')
    plt.ylabel('Gradients')
    plt.xticks(range(0, 5000, 1000), range(0, 50, 10))
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def activation_dist_per_epoch(acts, savepath=''):
    act1_mean = []
    act2_mean = []
    act3_mean = []
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for i, (k, v) in enumerate(acts.items()):
        act1_mean = np.array(v['act1_mean'])
        act2_mean = np.array(v['act2_mean'])
        act3_mean = np.array(v['act3_mean'])
        sns.distplot(act1_mean, label='Epoch {}'.format(i * 10), ax=ax1)
        sns.distplot(act2_mean, label='Epoch {}'.format(i * 10), ax=ax2)
        sns.distplot(act3_mean, label='Epoch {}'.format(i * 10), ax=ax3)
    ax1.set_ylabel('Counts')
    ax2.set_xlabel('Activation values')
    ax2.set_ylim(0, 1000)
    plt.legend()
    plt.show()
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0.1)


def load_epochs_grad_act(path, startswith):
    d = collections.OrderedDict()
    files = []
    for file in os.listdir(path):
        if file.startswith(startswith):
            files.append(file)
    files.sort()
    for file in files:
        d[file] = np.load(file, allow_pickle=True).item()
    return d


if __name__ == '__main__':
    # path = "./results/"
    # models = []
    # model_it = []
    # for file in os.listdir(path):
    #     if file.startswith('model'):
    #         models.append(torch.load(path + file))
    #         model_it.append(int(file.strip('model_it .pt')))
    # index = np.argsort(model_it)
    # model_it = np.array(model_it)[index]
    # models = np.array(models)[index]
    # weight_distributions_per_layer(
    #     models, model_it, 'conv1.weight', bins=10, savepath='sgd_weight_distributions_per_layer.eps')
    path = "./"
    # act_funct = np.load(path + 'act_func.npy', allow_pickle=True).item()
    # activation_functions_dist_iteration(act_funct, savepath=os.path.join(
    #    path, 'adam_activation_functions_dist_iteration.pdf'))
    # activation_functions_dist_layer(act_funct, savepath=os.path.join(
    #     path, 'adam_activation_functions_dist_layer.pdf'))
    # activation_functions_mean_std(act_funct, savepath=os.path.join(
    #     path, 'adam_activation_mean_std.pdf'))
    # gradients = np.load(path + 'gradients.npy', allow_pickle=True).item()
    # gradients_dist_layer(gradients, savepath=os.path.join(
    #     path, 'adam_grad_distribution_per_layer.pdf'))
    # gradients_mean_std(gradients, savepath=os.path.join(
    #     path, 'adam_grad_mean_std_all.pdf'))
    gradients = load_epochs_grad_act(path, 'gradients_ep')
    gradients_per_epoch(gradients, savepath='adam_grads_per_epoch.pdf')
    # acts = load_epochs_grad_act(path, 'act_func_ep')
    # activation_dist_per_epoch(
    #     acts, savepath='adam_activation_mean_std_per_epoch.pdf')
