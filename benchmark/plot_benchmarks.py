import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
# sns.set(style="white")
sns.set(style="white")
sns.set_color_codes("dark")
sns.set_context("paper", font_scale=1.5, rc={
                "lines.linewidth": 2., "grid.linewidth": 0.1})

def simple_timing(fn='timings_ep1.pt'):
    timings = torch.load(fn)
    index = ['Cpp', 'Cup', 'update', 'lstsq', 'mm']

    vs = []
    keys = []

    for i, (k, v) in enumerate(timings['1'].items()):
        if k in index:
            print(k)
        vs.append(np.sum(v))
        keys.append(k)

    y_pos = np.arange(len(keys))

    sns.barplot(vs, y_pos, orient='h', color='b')
    plt.yticks(y_pos, keys)
    plt.xlabel('times in [s]')
    plt.title('Timings in update step')
    plt.savefig(fn.split('.')[0], format='pdf')
    plt.show()


def multiple_ensemble_timings(filenames, index):
    fit_time = []
    x_label = []
    for fn in filenames:
        t = torch.load(fn)['1'][index]
        s = int(fn.split('_')[1].split('ens')[1])
        x_label.append(s)
        fit_time.append(sum(t))
    ax = sns.barplot(np.arange(len(x_label)), np.ravel(fit_time))

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + .1,
                '{:1.2f}'.format(height),
                ha="center")
    plt.xlabel('# ensembles')
    plt.xticks(range(len(x_label)), x_label)
    plt.ylabel('time in [s]')
    plt.show()


def to_dataframe(files):
    Cpps = []
    Cups = []
    updates = []
    fits = []
    model_outs = []
    x_label = []
    for f in files:
        dat = torch.load(f)
        Cpps.append(sum(dat['1']['Cpp']))
        Cups.append(sum(dat['1']['Cup']))
        updates.append(sum(dat['1']['update']))
        model_outs.append(sum(dat['1']['model_out']))
        fits.append(sum(dat['1']['fit']))
        s = int(f.split('_')[1].split('ens')[1])
        x_label.append(s)
    d = {'C(U)': np.unique(Cpps), r'D(U)': np.unique(Cups),
         'update': np.unique(updates), 'indexing': model_outs, 'fit': fits, 
         'ensembles': x_label}
    df = pd.DataFrame.from_dict(d)
    print(df)
    df = pd.melt(df, id_vars='ensembles',
                 var_name='functions', value_name='time in [s]')
    return df


files = ['timings_ens100_ep1.pt', 'timings_ens1000_ep1.pt',
         'timings_ens5000_ep1.pt', 'timings_ens10000_ep1.pt']
# multiple_ensemble_timings(files, 'fit')
data = to_dataframe(files)
fig = sns.catplot(x='functions', y='time in [s]', hue='ensembles',
                  data=data, kind='bar', palette="deep", legend=False, height=6)
fig.set(xlabel=None)
fig.fig.savefig('benchmarks.pdf', bbox_inches='tight', pad_inches=0.1)
# plt.show()
