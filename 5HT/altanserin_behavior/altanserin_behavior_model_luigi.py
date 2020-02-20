# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:40:23 2019

@author: guido
"""

from os.path import join, expanduser
from scipy.io import loadmat
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Settings
FIG_PATH = join(expanduser('~'), 'Figures', '5HT')

# Load in data
data = loadmat(join(expanduser('~'), 'Data', '5HT', 'guido_analysis_14feb2020.mat'))
runlength = data['X'][0]
parameters = data['pnames'][0]

# Get maximum probability density of fitted run length parameter
max_prob = np.zeros(np.size(runlength))
for i in range(np.size(runlength)):
    max_prob[i] = stats.mode(runlength[i][:, parameters == 'runlength-tau'])[0][0][0]

results = pd.DataFrame({'max_prob': max_prob,
                        'condition': [0, 1, 2]*int(np.size(runlength)/3),
                        'subject': np.repeat(np.arange(np.size(runlength)/3), 3)})

f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x='condition', y='max_prob', hue='subject', data=results, legend=False,
             ax=ax1, lw=3)
ax1.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Lenght of integration window (tau)',
        title='Fitted running average model', ylim=[1, 4])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine(trim=True)
plt.tight_layout(pad=2)

plt.savefig(join(FIG_PATH, '5HT2a_block_integration_lenght.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, '5HT2a_block_integration_lenght.png'), dpi=300)
