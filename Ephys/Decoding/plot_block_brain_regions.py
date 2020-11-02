#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode left/right block identity from all brain regions
@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import decode
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
import alf
from ephys_functions import paths, figure_style, combine_layers_cortex

# Settings
# METRIC = 'acc_over_chance'
METRIC = 'auroc'
MIN_PERF = 0.55
DECODER = 'bayes'
#MIN_PERF = -1
MIN_REC = 2
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')


# %% Plot

# Load in data
decoding_result = pd.read_csv(join(SAVE_PATH,
                                   ('decoding_block_regions_all_neurons_%s.csv' % DECODER)))

# Exclude root
decoding_result = decoding_result.reset_index()
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Drop duplicates
decoding_result = decoding_result[decoding_result.duplicated(subset=['region', 'eid', 'probe'])
                                  == False]

# Get decoding performance over chance
decoding_result['acc_over_chance'] = (decoding_result['accuracy']
                                      - decoding_result['chance_level']) * 100

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'acc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'acc_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'auroc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'auroc'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)

# Apply plotting threshold
decoding_result = decoding_result[(decoding_result['%s_mean' % METRIC] > MIN_PERF) &
                                  (decoding_result['n_rec'] >= MIN_REC)]

# Get sorting
sort_regions = decoding_result.groupby('region').mean().sort_values(METRIC, ascending=False).reset_index()['region']

figure_style(font_scale=2)
f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(x=METRIC, y='region', data=decoding_result,
            order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding accuracy of stimulus prior (% correct over chance level)', ylabel='',
        xlim=[0.5, 0.7])

plt.savefig(join(FIG_PATH, 'decode_block_combined_regions_all_neurons_%s' % DECODER))
