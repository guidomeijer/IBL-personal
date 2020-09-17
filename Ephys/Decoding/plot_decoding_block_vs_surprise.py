#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:31:04 2020

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from ephys_functions import paths, figure_style

# Settings
METRIC = 'auroc'
N_NEURONS = 10
SIDE = 'right'
_, FIG_PATH, SAVE_PATH = paths()

# Load in decoding results
decoding_block = pd.read_csv(join(SAVE_PATH, 'decoding_block_regions_%d_neurons.csv' % N_NEURONS))
decoding_surprise = pd.read_csv(join(SAVE_PATH, 'decoding_surprise_regions_%d_neurons.csv'
                                     % N_NEURONS))
decoding_surprise['f1_right_shuf'] = decoding_surprise['f1_right_shuffle']

# Exclude root
decoding_block = decoding_block.reset_index()
incl_regions = [i for i, j in enumerate(decoding_block['region']) if not j.islower()]
decoding_block = decoding_block.loc[incl_regions]
decoding_surprise = decoding_surprise.reset_index()
incl_regions = [i for i, j in enumerate(decoding_surprise['region']) if not j.islower()]
decoding_surprise = decoding_surprise.loc[incl_regions]

# Get block decoding results
block_regions = decoding_block['region'].unique()
decod_block = np.zeros(block_regions.shape)
for i, region in enumerate(block_regions):
    decod_result = (decoding_block.loc[decoding_block['region'] == region, METRIC]
                    - decoding_block.loc[decoding_block['region'] == region,
                                         '%s_shuffle' % METRIC]).mean()
    decod_block[i] = decod_result
    
# Get surprise decoding results
surprise_regions = decoding_surprise['region'].unique()
decod_surprise = np.zeros(surprise_regions.shape)
for i, region in enumerate(surprise_regions):
    decod_result = (decoding_surprise.loc[decoding_surprise['region'] == region,
                                          '%s_%s' % (METRIC, SIDE)]
                    - decoding_surprise.loc[decoding_surprise['region'] == region,
                                         '%s_%s_shuf' % (METRIC, SIDE)]).mean()
    decod_surprise[i] = decod_result
    
# Remove regions that are not in both analyses
decod_surprise = decod_surprise[[elem in block_regions for elem in surprise_regions]]
surprise_regions = surprise_regions[[elem in block_regions for elem in surprise_regions]]
    
f, ax1 = plt.subplots(1, 1, figsize=(6, 6))
ax1.scatter(decod_block, decod_surprise)