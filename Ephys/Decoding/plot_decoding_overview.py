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
import pandas as pd
import seaborn as sns
from ephys_functions import paths, figure_style, get_full_region_name, get_parent_region_name

# Settings
TARGET = 'reward'
DECODER = 'bayes-multinomial'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
CHANCE_LEVEL = 'shuffle'
VALIDATION = 'kfold-interleaved'
FULL_NAME = True
PARENT_REGIONS = False

# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                    INCL_SESSIONS, INCL_NEURONS))))

# Exclude root
decoding_result = decoding_result.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Drop duplicates
decoding_result = decoding_result[decoding_result.duplicated(subset=['region', 'eid', 'probe'])
                                  == False]

# %%

print('Number of insertions: %d' % decoding_result.groupby(['eid', 'probe']).size().shape[0])
print('Number of brain regions: %d' % decoding_result.groupby(['region']).size().shape[0])
print('Number of recordings in regions: %d' % decoding_result.shape[0])
print('Total number of cells: %d' % decoding_result['n_neurons'].sum())

figure_style(font_scale=1.8)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

ax1.hist(decoding_result['n_neurons'], bins=100)
ax1.set(title='Median number of neurons per region: %d' % decoding_result['n_neurons'].median(),
        xlim=[0, 300], xlabel='Number of neurons per region', ylabel='Count')

ax2.hist(decoding_result.groupby('region').size().values, bins=50)
ax2.set(title=('Median number of recordings per region: %d'
               % decoding_result.groupby('region').size().median()),
        xlabel='Number of recordings per region', ylabel='Count')

plt.tight_layout()
sns.despine(trim=False)

plt.savefig(join(FIG_PATH, 'decoding_overview'))
