#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:34:52 2020

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from ephys_functions import paths, figure_style

MIN_PERC = 10
MIN_REC = 2
MIN_NEURONS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
COMBINE_LAYERS_CORTEX = True

# %% Block neurons

if COMBINE_LAYERS_CORTEX:
    block_neurons = pd.read_csv(join(SAVE_PATH, 'n_block_neurons_combined_regions.csv'))
else:
    block_neurons = pd.read_csv(join(SAVE_PATH, 'n_block_neurons_regions.csv'))

block_neurons['perc'] = (block_neurons['n_sig_block'] / block_neurons['n_neurons']) * 100

# Exclude root
block_neurons = block_neurons.reset_index()
incl_regions = [i for i, j in enumerate(block_neurons['region']) if not j.islower()]
block_neurons = block_neurons.loc[incl_regions]

# Drop duplicates
block_neurons = block_neurons[~block_neurons.duplicated(subset=['region', 'eid', 'probe'])]

# Exclude regions with too few neurons
block_neurons = block_neurons[block_neurons['n_neurons'] >= MIN_NEURONS]

# Calculate average decoding performance per region
for i, region in enumerate(block_neurons['region'].unique()):
    block_neurons.loc[block_neurons['region'] == region, 'perc_mean'] = block_neurons.loc[
                            block_neurons['region'] == region, 'perc'].mean()
    block_neurons.loc[block_neurons['region'] == region, 'n_rec'] = np.sum(
                                                            block_neurons['region'] == region)

# Apply plotting threshold
block_neurons = block_neurons[(block_neurons['perc_mean'] > MIN_PERC) &
                                  (block_neurons['n_rec'] >= MIN_REC)]

# Get sorting
sort_block = block_neurons.groupby('region').mean().sort_values(
                                            'perc', ascending=False).reset_index()['region']

# %% Surprise neurons

if COMBINE_LAYERS_CORTEX:
    surprise_neurons = pd.read_csv(join(SAVE_PATH, 'n_surprise_neurons_combined_regions.csv'))
else:
    surprise_neurons = pd.read_csv(join(SAVE_PATH, 'n_surprise_neurons_regions.csv'))

surprise_neurons['perc'] = (surprise_neurons['n_sig_surprise']
                            / surprise_neurons['n_neurons']) * 100

# Exclude root
surprise_neurons = surprise_neurons.reset_index()
incl_regions = [i for i, j in enumerate(surprise_neurons['region']) if not j.islower()]
surprise_neurons = surprise_neurons.loc[incl_regions]

# Drop duplicates
surprise_neurons = surprise_neurons[~surprise_neurons.duplicated(
                                            subset=['region', 'eid', 'probe'])]

# Exclude regions with too few neurons
surprise_neurons = surprise_neurons[surprise_neurons['n_neurons'] >= MIN_NEURONS]

# Calculate average decoding performance per region
for i, region in enumerate(surprise_neurons['region'].unique()):
    surprise_neurons.loc[surprise_neurons['region'] == region, 'perc_mean'] = surprise_neurons.loc[
                            surprise_neurons['region'] == region, 'perc'].mean()
    surprise_neurons.loc[surprise_neurons['region'] == region, 'n_rec'] = np.sum(
                                                            surprise_neurons['region'] == region)

# Apply plotting threshold
surprise_neurons = surprise_neurons[(surprise_neurons['perc_mean'] > MIN_PERC) &
                                  (surprise_neurons['n_rec'] >= MIN_REC)]

# Get sorting
sort_surprise = surprise_neurons.groupby('region').mean().sort_values(
                                            'perc', ascending=False).reset_index()['region']

# %% Plot
figure_style(font_scale=1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), dpi=300)

sns.barplot(x='perc', y='region', data=block_neurons, order=sort_block, ci=68, ax=ax1)
ax1.set(xlabel='Stimulus prior neurons (%)', ylabel='')

sns.barplot(x='perc', y='region', data=surprise_neurons, order=sort_surprise, ci=68, ax=ax2)
ax2.set(xlabel='Stimulus inconsistency neurons (%)', ylabel='')

figure_style(despine=True)
plt.savefig(join(FIG_PATH, 'WholeBrain', 'block_surprise_neurons'), dpi=300)
